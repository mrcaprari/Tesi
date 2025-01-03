import math
import torch
import torch.fx
from ..distributions import Particle, Gaussian, RandomParameter, Prior

def add_functions_to_module(module, functions):
    for name, func in functions.items():
        setattr(module, name, func.__get__(module))
    return module


def named_particles(self):
    for name, submodule in self.named_modules():
        if isinstance(submodule, Particle):
            yield name, submodule.particles
        else:
            # For non-Particle modules, yield their parameters with an added dimension
            for param_name, param in submodule.named_parameters(recurse=False):
                # Add a new dimension and expand it to match self.n_particles
                expanded_param = param.unsqueeze(0).expand(self.n_particles, *param.size())
                yield f'{name}.{param_name}', expanded_param

def all_particles(self):
    return torch.cat([torch.flatten(tensor, start_dim=1) for _, tensor in self.named_particles()], dim=1)

def compute_kernel_matrix(self):
    particles = self.all_particles() #[n_particles, n_params]
    pairwise_sq_dists = torch.cdist(particles, particles, p=2) ** 2 
    median_squared_dist = pairwise_sq_dists.median()
    lengthscale = torch.sqrt(0.5 * median_squared_dist / math.log(self.n_particles))
    self.kernel_matrix = torch.exp(-pairwise_sq_dists/(2 * lengthscale ** 2))    

def perturb_gradients(self):
    self.compute_kernel_matrix()
    for name, param in self.named_parameters(recurse=True):
        # Remove the last word after the last dot
        parent_name = name.rsplit('.', 1)[0]
        # Check if the parent is a Particle submodule
        if isinstance(self.get_submodule(parent_name), Particle):
            self.get_submodule(parent_name).perturb_gradients(self.kernel_matrix)


def lazy_initialize_n_particles(module, n_particles):
    module.n_particles = n_particles
    # Remove the hook after it's executed
    if hasattr(module, '_lazy_hook'):
        module._lazy_hook.remove()
        del module._lazy_hook


def kl_divergence(self):
    kl_div = 0
    for name, submodule in self.named_modules():
        if isinstance(submodule, Gaussian) and not isinstance(submodule, Prior):
            var_ratio = (submodule.std / submodule.prior.std) ** 2
            current_kl = 0.5 * (torch.log(submodule.prior.std ** 2 / submodule.std ** 2) + var_ratio + 
                        ((submodule.mu - submodule.prior.mu) ** 2) / (submodule.prior.std ** 2) - 1)
            kl_div += current_kl.sum()
    return kl_div


