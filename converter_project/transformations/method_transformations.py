import math
from typing import Any, Callable, Dict, Generator, Tuple

import torch
from torch import Tensor
from torch.nn import Module

from ..distributions import Gaussian, Particle, Prior, RandomParameter


def add_functions_to_module(module: Module, functions: Dict[str, Callable]) -> Module:
    """
    Dynamically adds functions as methods to a module.

    Args:
        module (Module): The module to modify.
        functions (Dict[str, Callable]): A dictionary of function names and their implementations.

    Returns:
        Module: The modified module with added functions.
    """
    for name, func in functions.items():
        setattr(module, name, func.__get__(module))
    return module


def named_particles(self: Module) -> Generator[Tuple[str, Tensor], None, None]:
    """
    Generator yielding the names and particles or expanded parameters of a module.

    If a submodule is an instance of `Particle`, its particles are yielded. Otherwise,
    its parameters are expanded along a new dimension to match `self.n_particles`.

    Yields:
        Tuple[str, Tensor]: The name and tensor of particles or expanded parameters.
    """
    for name, submodule in self.named_modules():
        if isinstance(submodule, Particle):
            yield name, submodule.particles
        else:
            for param_name, param in submodule.named_parameters(recurse=False):
                expanded_param = param.unsqueeze(0).expand(
                    self.n_particles, *param.size()
                )
                yield f"{name}.{param_name}", expanded_param


def all_particles(self: Module) -> Tensor:
    """
    Concatenates all particles or expanded parameters into a single tensor.

    Returns:
        Tensor: A tensor of shape [n_particles, total_params], where `total_params`
        is the sum of flattened parameter sizes across all particles.
    """
    return torch.cat(
        [torch.flatten(tensor, start_dim=1) for _, tensor in self.named_particles()],
        dim=1,
    )


def compute_kernel_matrix(self: Module) -> None:
    """
    Computes the kernel matrix for particles based on pairwise squared distances.

    The kernel matrix is stored as `self.kernel_matrix`.

    Formula:
        kernel[i, j] = exp(-||x_i - x_j||^2 / (2 * lengthscale^2))

    The lengthscale is computed using the median heuristic.
    """
    particles = self.all_particles()  # Shape: [n_particles, n_params]
    pairwise_sq_dists = torch.cdist(particles, particles, p=2) ** 2
    median_squared_dist = pairwise_sq_dists.median()
    lengthscale = torch.sqrt(0.5 * median_squared_dist / math.log(self.n_particles))
    self.kernel_matrix = torch.exp(-pairwise_sq_dists / (2 * lengthscale**2))


def perturb_gradients(self: Module) -> None:
    """
    Perturbs gradients of particle-based parameters using the kernel matrix.

    Requires `self.kernel_matrix` to be computed prior to calling this method.
    """
    self.compute_kernel_matrix()
    for particle in self.particle_modules:
        particle.perturb_gradients(self.kernel_matrix)


def initialize_particles(module: Module, n_particles: int) -> None:
    """
    Initializes a module for particle-based computations.

    Adds `n_particles` and `particle_modules` attributes to the module.

    Args:
        module (Module): The module to initialize.
        n_particles (int): The number of particles.
    """
    module.n_particles = n_particles
    module.particle_modules = [
        submodule for submodule in module.modules() if isinstance(submodule, Particle)
    ]


def kl_divergence(self: Module) -> Tensor:
    """
    Computes the Kullback-Leibler (KL) divergence between parameters and their priors.

    For each submodule of type `Gaussian` (excluding `Prior`), computes:
        KL = 0.5 * (log(std_prior^2 / std^2) + (std^2 / std_prior^2) +
                    ((mu - mu_prior)^2 / std_prior^2) - 1)

    Returns:
        Tensor: The total KL divergence across all parameters.
    """
    kl_div = 0
    for name, submodule in self.named_modules():
        if isinstance(submodule, Gaussian) and not isinstance(submodule, Prior):
            var_ratio = (submodule.std / submodule.prior.std) ** 2
            current_kl = 0.5 * (
                torch.log(submodule.prior.std**2 / submodule.std**2)
                + var_ratio
                + ((submodule.mu - submodule.prior.mu) ** 2) / (submodule.prior.std**2)
                - 1
            )
            kl_div += current_kl.sum()
    return kl_div
