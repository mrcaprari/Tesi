import math
from typing import Any, Generator, List, Optional, Tuple

import torch
import torch.fx
from torch.fx import GraphModule

from converter_project.converters.graphmodule_builders import (
    GaussianGraphModuleBuilder,
    ParticleGraphModuleBuilder,
)
from converter_project.converters.tracers import PreparatoryTracer
from converter_project.converters.transformers import VmapTransformer
from converter_project.distributions import Gaussian, Particle, Prior


class Converter:
    def __init__(
        self,
        tracer=None,
        graph_module_builder=None,
        transformer=None,
        toplevel_methods=None,
    ):
        self.tracer = tracer
        self.graph_module_builder = graph_module_builder
        self.transformer = transformer
        self.toplevel_methods = toplevel_methods or {}

    def convert(
        self, module: torch.nn.Module, parameter_list: Optional[List[Any]] = []
    ) -> GraphModule:
        original_graph = self.tracer().trace(module, parameter_list)
        new_module, new_graph = self.graph_module_builder(
            original_graph
        ).build_graph_module(module)

        transformed_module = GraphModule(new_module, new_graph)

        final_module = self.transformer(transformed_module).transform()

        final_module = self.add_methods(final_module)

        return final_module

    def add_methods(self, module: GraphModule) -> GraphModule:
        for method_name, method_func in self.toplevel_methods.items():
            # Dynamically add the method to the module
            setattr(module, method_name, method_func.__get__(module, type(module)))
        return module


class ParticleConverter(Converter):
    def __init__(
        self,
        tracer=PreparatoryTracer,
        graph_module_builder=ParticleGraphModuleBuilder,
        transformer=VmapTransformer,
        toplevel_methods={},
    ):
        if toplevel_methods is None:
            toplevel_methods = {}
        # Add particle-related methods to `toplevel_methods`
        toplevel_methods.update(
            {
                "named_particles": self.named_particles,
                "all_particles": self.all_particles,
                "compute_kernel_matrix": self.compute_kernel_matrix,
                "perturb_gradients": self.perturb_gradients,
                "initialize_particles": self.initialize_particles,
            }
        )
        super().__init__(tracer, graph_module_builder, transformer, toplevel_methods)


class ParticleConverter(Converter):
    def __init__(
        self,
        tracer=PreparatoryTracer,
        graph_module_builder=ParticleGraphModuleBuilder,
        transformer=VmapTransformer,
        toplevel_methods=None,
    ):
        if toplevel_methods is None:
            toplevel_methods = {}
        # Add particle-related methods to `toplevel_methods`
        toplevel_methods.update(
            {
                "named_particles": self._named_particles,
                "all_particles": self._all_particles,
                "compute_kernel_matrix": self._compute_kernel_matrix,
                "perturb_gradients": self._perturb_gradients,
                "initialize_particles": self._initialize_particles,
            }
        )
        super().__init__(tracer, graph_module_builder, transformer, toplevel_methods)

    @staticmethod
    def _initialize_particles(module: torch.nn.Module, n_particles: int) -> None:
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
        return torch.cat(
            [
                torch.flatten(tensor, start_dim=1)
                for _, tensor in module.named_particles()
            ],
            dim=1,
        )

    @staticmethod
    def _compute_kernel_matrix(module: torch.nn.Module) -> None:
        particles = module.all_particles()  # Shape: [n_particles, n_params]
        pairwise_sq_dists = torch.cdist(particles, particles, p=2) ** 2
        median_squared_dist = pairwise_sq_dists.median()
        lengthscale = torch.sqrt(
            0.5 * median_squared_dist / math.log(module.n_particles)
        )
        module.kernel_matrix = torch.exp(-pairwise_sq_dists / (2 * lengthscale**2))

    @staticmethod
    def _perturb_gradients(module: torch.nn.Module) -> None:
        module.compute_kernel_matrix()
        for particle in module.particle_modules:
            particle.perturb_gradients(module.kernel_matrix)


class GaussianConverter(Converter):
    def __init__(
        self,
        tracer=PreparatoryTracer,
        graph_module_builder=GaussianGraphModuleBuilder,
        transformer=VmapTransformer,
        toplevel_methods={},
    ):
        toplevel_methods.update(
            {
                "kl_divergence": self._kl_divergence,
            }
        )
        super().__init__(tracer, graph_module_builder, transformer, toplevel_methods)

    @staticmethod
    def _kl_divergence(module: torch.nn.Module) -> torch.Tensor:
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
