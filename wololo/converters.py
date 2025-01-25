import math
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

import torch
import torch.fx
from torch.fx import GraphModule

from .adapters import Adapter
from .tracers import PreparatoryTracer
from .transformers import VmapTransformer


class Converter:
    """
    Manages the transformation of a deterministic model into a stochastic model by:
    - Tracing and preparing the computation graph of the deterministic model.
    - Adapting parameters and nodes stochasticly, accounting for multiple realizations.
    - Transforming the forward logic to support batched parameter dimensions.

    Attributes:
        tracer (torch.fx.Tracer): Traces the computation graph. Defaults to `PreparatoryTracer`,
            which adds "transform" metadata to nodes for stochastic adaptation; user can change this
            to allow for tracing dynamic architectures.
        adapter (Adapter): Adapts the graph and substitutes parameters with stochastic modules.
            Defaults to `Adapter`.
        transformer (torch.fx.Transformer): Transforms the forward method. Defaults to `VmapTransformer`,
            well suited for dense architectures; users can change this for more optimized forward() transformations.
        toplevel_methods (Dict): A dictionary of methods to be added to the transformed module,
            useful for implementing posterior approximation algorithms like BBVI or SVGD.
    """

    def __init__(
        self,
        tracer: torch.fx.Tracer = None,
        adapter: Adapter = None,
        transformer: torch.fx.Transformer = None,
        toplevel_methods: Dict = None,
    ) -> None:
        """
        Initializes the Converter with configurable components.

        Args:
            tracer (torch.fx.Tracer, optional): A tracer for graph preparation.
                Defaults to `PreparatoryTracer`.
            adapter (Adapter, optional): Handles stochastic adaptation of parameters and nodes.
                Defaults to `Adapter`.
            transformer (torch.fx.Transformer, optional): Handles forward method transformation.
                Defaults to `VmapTransformer`.
            toplevel_methods (Dict, optional): Methods to add at the top level of the transformed module.
        """
        self.tracer = tracer or PreparatoryTracer
        self.adapter = adapter or Adapter
        self.transformer = transformer or VmapTransformer
        self.toplevel_methods = toplevel_methods or {}

    def convert(
        self,
        module: torch.nn.Module,
        stochastic_parameter: type[torch.nn.Module],
        parameter_list: Optional[list] = None,
        **kwargs,
    ) -> GraphModule:
        """
        Converts a module by applying the three-part transformation: tracing, stochastic adaptation,
        and forward method transformation. Adds methods specified in `toplevel_methods` to the final module.

        Args:
            module (torch.nn.Module): The module to be transformed.
            stochastic_parameter (type[torch.nn.Module]): A class representing stochastic parameters.
                The `forward()` method of this class must return realizations of the parameter
                and accept `n_samples` as input to generate multiple realizations.
            parameter_list (Optional[List[str]], optional): List of parameter names to replace stochastically.
                Defaults to all parameters if not specified.

        Returns:
            GraphModule: The transformed module with stochastic parameters and an updated computation graph.

        Notes:
            - For dense architectures, specify the `stochastic_parameter` class, and Converter
              handles stochastic adaptation automatically.
            - For dynamic architectures, customize the `tracer` to accommodate for
              different graph preparation or transformation approaches.
            - For convolutional architectures, customize the `transformer` to accommodate for
              different transformation approaches.
        """
        if parameter_list is None:
            parameter_list = []

        original_graph = self.tracer().trace(module, parameter_list)
        new_module, new_graph = self.adapter(original_graph).adapt_module(
            module, stochastic_parameter, **kwargs
        )

        transformed_module = GraphModule(new_module, new_graph)
        final_module = self.transformer(transformed_module).transform()

        self.add_methods(final_module, stochastic_parameter)
        return final_module

    def add_methods(self, module: GraphModule, stochastic_parameter) -> None:
        """
        Adds user-defined methods to the transformed module.

        Args:
            module (GraphModule): The module to which methods will be added.
        """
        module.stochastic_parameters = [
            (name, module)
            for name, module in module.named_modules()
            if isinstance(module, stochastic_parameter)
        ]

        for method_name, method_function in self.toplevel_methods.items():
            setattr(module, method_name, method_function.__get__(module, type(module)))


class BBVIConverter(Converter):
    def __init__(
        self,
        tracer: torch.fx.Tracer = None,
        adapter: Adapter = None,
        transformer: torch.fx.Transformer = None,
        toplevel_methods: Dict = None,
    ) -> None:

        toplevel_methods = toplevel_methods or {}
        toplevel_methods.update({"kl_divergence": self._kl_divergence})
        super().__init__(tracer, adapter, transformer, toplevel_methods)

    def convert(
        self,
        module: torch.nn.Module,
        stochastic_parameter: type[torch.nn.Module],
        parameter_list: Optional[list] = None,
        **kwargs,
    ) -> GraphModule:

        if not hasattr(stochastic_parameter, "kl_divergence"):
            raise ValueError(
                "stochastic_parameter must have a kl_divergence method for BBVI"
            )

        return super().convert(module, stochastic_parameter, parameter_list, **kwargs)

    def _kl_divergence(self) -> torch.Tensor:
        kl_div = 0
        for name, stochastic_parameter in self.stochastic_parameters:
            kl_div += stochastic_parameter.kl_divergence()
        return kl_div


class ParticleParameter(torch.nn.modules.lazy.LazyModuleMixin, torch.nn.Module):
    def __init__(self, parameter, particle_config={"prior_std": 1.0}):
        super().__init__()
        self.particle_config = particle_config

        if "prior" in self.particle_config:
            self.prior = self.particle_cofig["prior"]
        else:
            self.prior = torch.distributions.Normal(
                loc=parameter,
                scale=torch.full(parameter.size(), self.particle_config["prior_std"]),
            )
        self.register_parameter("particles", torch.nn.UninitializedParameter())

    def initialize_parameters(self, n_samples: int) -> None:

        if self.has_uninitialized_params():
            self.particles.materialize((n_samples, *self.prior.loc.shape))

        with torch.no_grad():
            self.particles = torch.nn.Parameter(
                self.prior.rsample(
                    (n_samples,),
                )
            )

        suffix = "".join(
            [chr(ord("k") + i) for i in range(self.particles.ndim - 1)]
        )  # Dynamically compute suffix
        self.einsum_equation = f"ij,j{suffix}->i{suffix}"  # Precompute einsum equation

    @property
    def flattened_particles(self) -> torch.Tensor:
        return torch.flatten(self.particles, start_dim=1)

    def perturb_gradients(self, kernel_matrix: torch.Tensor) -> None:
        # Direct einsum operation using the precomputed einsum equation
        self.particles.grad = torch.einsum(
            self.einsum_equation, kernel_matrix, self.particles.grad
        )

    def forward(self, n_samples):
        return self.particles


class SVGDConverter(Converter):
    def __init__(
        self,
        tracer: torch.fx.Tracer = None,
        adapter: Adapter = None,
        transformer: torch.fx.Transformer = None,
        toplevel_methods: Dict = None,
    ) -> None:

        toplevel_methods = toplevel_methods or {}
        toplevel_methods.update(
            {
                "named_particles": self.named_particles,
                "all_particles": self.all_particles,
                "compute_kernel_matrix": self.compute_kernel_matrix,
                "perturb_gradients": self.perturb_gradients,
            }
        )
        super().__init__(tracer, adapter, transformer, toplevel_methods)

    def convert(
        self,
        module: torch.nn.Module,
        stochastic_parameter: type[torch.nn.Module] = ParticleParameter,
        parameter_list: Optional[list] = None,
        **kwargs,
    ) -> GraphModule:

        transformed_module = super().convert(
            module, stochastic_parameter, parameter_list, **kwargs
        )

        transformed_module.particle_modules = [
            submodule
            for submodule in transformed_module.modules()
            if isinstance(submodule, ParticleParameter)
        ]
        return transformed_module

    def named_particles(self):
        """
        Yield named particles for all `ParticleParam` submodules.

        Yields:
            Tuple[str, torch.Tensor]: Name and tensor of particles.
        """
        for name, submodule in self.named_modules():
            if isinstance(submodule, ParticleParameter):
                yield name, submodule.particles
            else:
                for param_name, param in submodule.named_parameters(recurse=False):
                    expanded_param = param.unsqueeze(0).expand(
                        self.n_particles, *param.size()
                    )
                    yield f"{name}.{param_name}", expanded_param

    def all_particles(self) -> torch.Tensor:
        """
        Concatenate all particles into a single tensor.

        Returns:
            torch.Tensor: Flattened and concatenated particles from all submodules.
        """
        return torch.cat(
            [
                torch.flatten(tensor, start_dim=1)
                for _, tensor in self.named_particles()
            ],
            dim=1,
        )

    def compute_kernel_matrix(self) -> None:
        """
        Computes the RBF kernel matrix for the particles in the model.
        """
        particles = self.all_particles()
        n_particles = particles.shape[0]
        pairwise_sq_dists = torch.cdist(particles, particles, p=2) ** 2
        median_squared_dist = pairwise_sq_dists.median()

        lengthscale = torch.sqrt(
            0.5
            * median_squared_dist
            / (torch.log(torch.tensor(n_particles, dtype=particles.dtype)) + 1e-8)
        )
        self.kernel_matrix = torch.exp(-pairwise_sq_dists / (2 * lengthscale**2))

    def perturb_gradients(self) -> None:
        """
        Adjust gradients of all particles in the model using the kernel matrix.
        """
        self.compute_kernel_matrix()
        for particle in self.particle_modules:
            particle.perturb_gradients(self.kernel_matrix)
