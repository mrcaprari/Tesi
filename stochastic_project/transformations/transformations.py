import torch
import torch.fx
from ..distributions import DistributionFactory, Gaussian, Prior, RandomParameter
from abc import ABC, abstractmethod
from .forward_transformations import vmap_forward_transformer
from .node_transformations import gaussian_mean_node, random_sample_node
from .parameter_transformations import to_gaussian, to_particle, to_standard_gaussian

class BaseTransformation:
    def transform_parameter(self, param):
        """Default: Return the parameter unchanged."""
        return param

    def transform_node(self, new_node, old_node, n_samples_node, val_map):
        """Default: Copy the node without modifications."""
        return new_node

    def transform_forward(self, module):
        """Default: Return the module unchanged."""
        return module


class GaussianTransformation(BaseTransformation):
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
