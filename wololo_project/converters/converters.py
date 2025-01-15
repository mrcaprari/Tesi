import math
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

import torch
import torch.fx
from torch.fx import GraphModule

from ..distributions import Gaussian, Particle, Prior
from .adapters import Adapter, GaussianAdapter, ParticleAdapter
from .tracers import PreparatoryTracer
from .transformers import VmapTransformer


class Converter:
    """
    The `Converter` class is the core component of the library, responsible for transforming PyTorch modules
    by replacing `torch.nn.Parameter` instances with custom `torch.nn.Module` instances. This transformation
    preserves the computation graph structure while enabling custom modifications.

    Components:
        - `tracer`: A specialized subclass of `torch.fx.Tracer`, such as `PreparatoryTracer`, used to capture
          and prepare the computation graph. It ensures detailed breakdown of nodes and provides metadata
          for identifying and targeting specific parameters during transformation.
        - `adapter`: A custom implementation of an `Adapter`, responsible for replacing
          parameters with custom modules and updating computation graph nodes to ensure consistency.
        - `transformer`: A subclass of `torch.fx.Transformer`, such as `VmapTransformer`, that modifies the
          `forward()` method of the transformed module for enhanced functionality, such as batched computation.
        - `toplevel_methods`: A dictionary of additional methods that are dynamically added to the top-level
          module for extended functionality.

    Attributes:
        tracer (torch.fx.Tracer): A `torch.fx.Tracer` instance for tracing and preparing the computational graph of a module.
        adapter (Adapter):  An instance of `Adapter`, responsible for replacing parameters with
                                                    custom modules and updating the computation graph to maintain
                                                    consistency. This includes modifying graph nodes like `get_attr` to
                                                    point to the new parameter or its associated methods.
        transformer (torch.fx.Transformer): A specialized `torch.fx.Transformer`, such as `VmapTransformer`, that adjusts
                                           the `forward()` method of the transformed module. The transformer ensures
                                           compatibility with new parameter structures and enables efficient batched
                                           computations.
        toplevel_methods (Dict[str, Callable]): A dictionary of method names to their implementations,
                                                which will be added to the transformed graph module. Defaults to an empty
                                                dictionary if not provided.
    """

    def __init__(
        self,
        tracer: torch.fx.Tracer,
        adapter: Adapter,
        transformer: torch.fx.Transformer,
        toplevel_methods: Dict[str, Callable[..., Any]] = None,
    ) -> None:
        """
        Initialize the Converter with its essential components.

        Args:
            tracer (Tracer): A specialized `torch.fx.Tracer` instance, such as `PreparatoryTracer`, responsible for tracing
                             the computational graph. The tracer prepares a fully detailed representation of
                             the computation graph, enabling targeted modifications during conversion.
            adapter (Callable): An instance of `Adapter`, responsible for
                                            replacing parameters with custom modules and updating the computation
                                            graph to maintain consistency. This includes modifying graph nodes like
                                            `get_attr` to point to the new parameter or its associated methods.
            transformer (Transformer): A specialized `torch.fx.Transformer`, such as `VmapTransformer`, that adjusts
                                       the `forward()` method of the transformed module. The transformer ensures
                                       compatibility with new parameter structures and enables efficient batched
                                       computations.
            toplevel_methods (Dict[str, Callable]): A dictionary of custom methods to be dynamically added to the
                                                    top-level module after transformation. Defaults to an empty
                                                    dictionary if not provided.
        """
        self.tracer = tracer
        self.adapter = adapter
        self.transformer = transformer
        self.toplevel_methods = toplevel_methods or {}

    def convert(
        self, module: torch.nn.Module, parameter_list: Optional[list] = None
    ) -> GraphModule:
        """
        Convert a PyTorch module into a `GraphModule` with dynamically added methods.

        The process involves tracing the module to capture its computation graph, replacing parameters with
        custom modules, and transforming the `forward()` method to accommodate the new parameter structure.

        Args:
            module (torch.nn.Module): The module to be converted.
            parameter_list (Optional[List[Any]]): A list of parameters to guide tracing and indicate which
                                                  parameters should be replaced. If no list is provided,
                                                  all parameters in the module will be transformed.

        Returns:
            GraphModule: The transformed graph module with additional methods dynamically added.
        """
        if parameter_list is None:
            parameter_list = []

        original_graph = self.tracer().trace(module, parameter_list)
        new_module, new_graph = self.adapter(original_graph).adapt_module(module)

        transformed_module = GraphModule(new_module, new_graph)
        final_module = self.transformer(transformed_module).transform()
        return self.add_methods(final_module)

    def add_methods(self, module: GraphModule) -> GraphModule:
        """
        Dynamically add methods to a `GraphModule` instance.

        The provided `toplevel_methods` dictionary contains method names and their implementations, which
        are attached to the top-level module during this step.

        Args:
            module (GraphModule): The graph module to which methods will be added.

        Returns:
            GraphModule: The updated graph module with new methods.
        """
        for method_name, method_func in self.toplevel_methods.items():
            setattr(module, method_name, method_func.__get__(module, type(module)))
        return module


class GaussianConverter(Converter):
    """
    A specialized converter for PyTorch modules, transforming them into probabilistic models with
    Gaussian random parameters. This transformation enables the models to perform Variational
    Inference by incorporating a `kl_divergence` method for computing the Kullback-Leibler (KL)
    divergence.

    Purpose:
        The `GaussianConverter` replaces standard `torch.nn.Parameter` instances in a module with
        custom Gaussian distributions. These distributions act as random variables, allowing for
        probabilistic modeling and Bayesian learning. This setup is essential for performing
        Variational Inference, as the `kl_divergence` method provides a way to compute the divergence
        between the posterior and prior distributions of these parameters.

    Attributes:
        tracer (torch.fx.Tracer): A `torch.fx.Tracer` instance for capturing and preparing the computation
                         graph of the module. Defaults to `PreparatoryTracer`.
        adapter (Adapter): A callable for building a new graph module from the traced
                                         computation graph. Defaults to `GaussianAdapter`.
        transformer (torch.fx.Transformer): A `torch.fx.Transformer` instance for modifying the `forward()`
                                   method to accommodate Gaussian random parameters. Defaults to
                                   `VmapTransformer`.
        toplevel_methods (Dict[str, Callable]): A dictionary of method names and their corresponding
                                                implementations, dynamically added to the transformed
                                                module. Includes the `kl_divergence` method for Variational
                                                Inference.
    """

    def __init__(
        self,
        tracer: torch.fx.Tracer = PreparatoryTracer,
        adapter: Adapter = GaussianAdapter,
        transformer: torch.fx.Transformer = VmapTransformer,
        toplevel_methods: Optional[Dict[str, Callable[..., Any]]] = None,
    ) -> None:
        """
        Initialize the GaussianConverter.

        Args:
            tracer (torch.fx.Tracer): A `torch.fx.Tracer` instance for capturing the module's computation graph.
                             Defaults to `PreparatoryTracer`.
            adapter (Adapter): A callable for building a graph module from the traced graph,
                                             replacing parameters with Gaussian random variables.
                                             Defaults to `GaussianAdapter`.
            transformer (torch.fx.Transformer): A `torch.fx.Transformer` instance for modifying the computation graph
                                       to support Gaussian random parameters and their behavior in the model.
                                       Defaults to `VmapTransformer`.
            toplevel_methods (Dict[str, Callable]): Additional methods to add to the converted module.
                                                    Defaults to a dictionary containing the `kl_divergence` method.
        """
        if toplevel_methods is None:
            toplevel_methods = {}
        toplevel_methods.update(
            {
                "kl_divergence": self._kl_divergence,
            }
        )
        super().__init__(tracer, adapter, transformer, toplevel_methods)

    @staticmethod
    def _kl_divergence(module: torch.nn.Module) -> torch.Tensor:
        """
        A method to be added to the transformed graph.
        It allows for computing the KL divergence for all Gaussian random parameters in the module.

        The KL divergence is calculated for each Gaussian parameter in the module as the
        divergence between its posterior and prior distributions. The total KL divergence
        is the sum across all such parameters.

        Args:
            module (torch.nn.Module): The module containing Gaussian distributions as parameters.

        Returns:
            torch.Tensor: The total KL divergence across all Gaussian distributions in the module.
        """
        kl_div = 0
        for name, submodule in module.named_modules():
            if isinstance(submodule, Gaussian) and not isinstance(submodule, Prior):
                var_ratio = (submodule.std / submodule.prior.std) ** 2
                current_kl = 0.5 * (
                    torch.log(submodule.prior.std**2 / submodule.std**2)
                    + var_ratio
                    + ((submodule.mu - submodule.prior.mu) ** 2)
                    / (submodule.prior.std**2)
                    - 1
                )
                kl_div += current_kl.sum()
        return kl_div


class ParticleConverter(Converter):
    """
    A specialized converter for PyTorch modules, transforming them into probabilistic models with
    multiple particle representations of their parameters. This transformation enables the models to
    leverage Stein Variational Gradient Descent (SVGD) by adding methods for kernel computation and
    suitable gradient perturbation.

    Purpose:
        The `ParticleConverter` replaces standard parameters with a collection of particles, where
        each particle represents a sample from a parameter's posterior distribution. This setup
        is integral for performing Bayesian inference with SVGD. Additional methods, such as
        `compute_kernel_matrix` and `perturb_gradients`, are dynamically added to the module,
        making it suitable for SVGD by modifying gradients based on a kernel.

    Attributes:
        tracer (torch.fx.Tracer): A `torch.fx.Tracer` instance for capturing and preparing the computation
                         graph of the module. Defaults to `PreparatoryTracer`.
        adapter (Adapter): A callable for building a new graph module from the traced
                                         computation graph. Defaults to `ParticleAdapter`.
        transformer (torch.fx.Transformer): A `torch.fx.Transformer` instance for modifying the `forward()`
                                   method to accommodate particle representations. Defaults to
                                   `VmapTransformer`.
        toplevel_methods (Dict[str, Callable]): A dictionary of method names and their corresponding
                                                implementations, dynamically added to the transformed
                                                module. Includes particle-specific methods for kernel
                                                computation and gradient perturbation.
    """

    def __init__(
        self,
        tracer: torch.fx.Tracer = PreparatoryTracer,
        adapter: Adapter = ParticleAdapter,
        transformer: torch.fx.Transformer = VmapTransformer,
        toplevel_methods: Optional[Dict[str, Callable[..., Any]]] = None,
    ) -> None:
        """
        Initialize the ParticleConverter.

        Args:
            tracer (torch.fx.Tracer): A `torch.fx.Tracer` instance for capturing the module's computation graph.
                             Defaults to `PreparatoryTracer`.
            adapter (Adapter): A callable for building a graph module from the traced graph,
                                             replacing parameters with particle representations.
                                             Defaults to `ParticleAdapter`.
            transformer (torch.fx.Transformer): A `torch.fx.Transformer` instance for modifying the computation graph
                                       to handle multiple particle representations.
                                       Defaults to `VmapTransformer`.
            toplevel_methods (Dict[str, Callable]): Additional methods to add to the converted module.
                                                    Defaults to a dictionary containing particle-specific methods.
        """
        if toplevel_methods is None:
            toplevel_methods = {}
        toplevel_methods.update(
            {
                "named_particles": self._named_particles,
                "all_particles": self._all_particles,
                "compute_kernel_matrix": self._compute_kernel_matrix,
                "perturb_gradients": self._perturb_gradients,
                "initialize_particles": self._initialize_particles,
            }
        )
        super().__init__(tracer, adapter, transformer, toplevel_methods)

    @staticmethod
    def _initialize_particles(module: torch.nn.Module, n_particles: int) -> None:
        """
        Initialize the particle-specific properties of a module.

        This method replaces standard parameters in the module with a specified number of
        particle representations, enabling Bayesian modeling.

        Args:
            module (torch.nn.Module): The module to initialize particles for.
            n_particles (int): The number of particles to initialize.
        """
        module.n_particles = n_particles
        module.particle_modules = [
            submodule
            for submodule in module.modules()
            if isinstance(submodule, Particle)
        ]

    @staticmethod
    def _named_particles(
        module: torch.nn.Module,
    ) -> Generator[Tuple[str, torch.Tensor], None, None]:
        """
        Generate named particles and their corresponding tensors in the module.

        Args:
            module (torch.nn.Module): The module to extract particles from.

        Yields:
            Tuple[str, torch.Tensor]: A tuple containing the name and tensor of each particle.
        """
        for name, submodule in module.named_modules():
            if isinstance(submodule, Particle):
                yield name, submodule.particles
            else:
                for param_name, param in submodule.named_parameters(recurse=False):
                    expanded_param = param.unsqueeze(0).expand(
                        module.n_particles, *param.size()
                    )
                    yield f"{name}.{param_name}", expanded_param

    @staticmethod
    def _all_particles(module: torch.nn.Module) -> torch.Tensor:
        """
        Concatenate all particle tensors in the module.

        This method aggregates all particle representations into a single tensor for
        efficient computation.

        Args:
            module (torch.nn.Module): The module containing particles.

        Returns:
            torch.Tensor: A tensor containing all particles concatenated along dimension 1.
        """
        return torch.cat(
            [
                torch.flatten(tensor, start_dim=1)
                for _, tensor in module.named_particles()
            ],
            dim=1,
        )

    @staticmethod
    def _compute_kernel_matrix(module: torch.nn.Module) -> None:
        """
        Compute the RBF kernel matrix for the particles in the module.

        Args:
            module (torch.nn.Module): The module containing particles.
        """
        particles = module.all_particles()  # Shape: [n_particles, n_params]
        pairwise_sq_dists = torch.cdist(particles, particles, p=2) ** 2
        median_squared_dist = pairwise_sq_dists.median()
        lengthscale = torch.sqrt(
            0.5 * median_squared_dist / math.log(module.n_particles)
        )
        module.kernel_matrix = torch.exp(-pairwise_sq_dists / (2 * lengthscale**2))

    @staticmethod
    def _perturb_gradients(module: torch.nn.Module) -> None:
        """
        Apply gradient perturbations to the particles in the module.

        This method adjusts the gradients of each particle using the computed kernel matrix,
        as required by the SVGD update rule.

        Args:
            module (torch.nn.Module): The module containing particles.
        """
        module.compute_kernel_matrix()
        for particle in module.particle_modules:
            particle.perturb_gradients(module.kernel_matrix)
