import torch
import torch.fx
from ..distributions import DistributionFactory, Gaussian, Prior, RandomParameter, Particle
from abc import ABC, abstractmethod
from .forward_transformations import vmap_forward_transformer
from .node_transformations import gaussian_mean_node, random_sample_node
from .parameter_transformations import to_gaussian, to_particle, to_standard_gaussian
from .method_transformations import add_functions_to_module, named_particles, all_particles, compute_kernel_matrix, kl_divergence, perturb_gradients

class BaseTransformation:
    def __init__(self, methods = {}, custom_methods = {}):
        super().__init__()
        self.methods = methods
        if custom_methods is not None:
            # Ensure all values in other_methods are callable
            if not all(callable(func) for func in custom_methods.values()):
                raise ValueError("All values in other_methods must be callable functions.")
            self.methods.update(custom_methods)
    
    def transform_parameter(self, param):
        """Default: Return the parameter unchanged."""
        return param

    def transform_node(self, new_node, old_node, n_samples_node, val_map):
        """Default: Copy the node without modifications."""
        return new_node

    def transform_forward(self, module):
        """Default: Return the module unchanged."""
        return module

    def add_methods(self, module):
        return add_functions_to_module(module, self.methods)

class GaussianTransformation(BaseTransformation):
    def __init__(self, custom_methods = {}):
        self.methods ={
            'kl_divergence': kl_divergence,
            }
        super().__init__(self.methods, custom_methods)
    
    def transform_parameter(self, param):
        return to_gaussian(param)

    def transform_node(self, new_node, old_node, n_samples_node, val_map):
        return random_sample_node(new_node, old_node, n_samples_node, val_map)

    def transform_forward(self, module):
        return vmap_forward_transformer(module)


class GaussianDeterministicTransformation(BaseTransformation):
    def transform_parameter(self, param):
        return to_gaussian(param)

    def transform_node(self, new_node, old_node, n_samples_node, val_map):
        return gaussian_mean_node(new_node, old_node, n_samples_node, val_map)

    def transform_forward(self, module):
        return vmap_forward_transformer(module)


class ParticleTransformation(BaseTransformation):
    def __init__(self, custom_methods = {}):
        self.methods ={
            'named_particles': named_particles,
            'all_particles': all_particles,
            'compute_kernel_matrix': compute_kernel_matrix,
            'perturb_gradients': perturb_gradients,
            }
        super().__init__(self.methods, custom_methods)

    def transform_parameter(self, param):
        return to_particle(param)

    def transform_node(self, new_node, old_node, n_samples_node, val_map):
        return random_sample_node(new_node, old_node, n_samples_node, val_map)

    def transform_forward(self, module):
        return vmap_forward_transformer(module)
