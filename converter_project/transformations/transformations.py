from typing import Any, Callable, Dict, Optional

from .forward_transformations import *
from .method_transformations import *
from .node_transformations import *
from .parameter_transformations import *


class Transformation:
    def __init__(
        self,
        base_methods: Optional[Dict[str, Callable]] = None,
        custom_methods: Optional[Dict[str, Callable]] = None,
    ) -> None:
        """
        Base class for defining transformations. Allows adding methods to the transformed module.

        Args:
            base_methods (Optional[Dict[str, Callable]]): Default transformation methods.
            custom_methods (Optional[Dict[str, Callable]]): Additional user-defined methods.
        """
        self.methods: Dict[str, Callable] = base_methods or {}
        if custom_methods:
            if not all(callable(func) for func in custom_methods.values()):
                raise ValueError("All values in custom_methods must be callable.")
            self.methods.update(custom_methods)

    def transform_parameter(self, param: Any) -> Any:
        """
        Applies a transformation to a parameter. Default implementation returns the parameter unchanged.

        Args:
            param (Any): The parameter to transform.

        Returns:
            Any: The transformed parameter.
        """
        return param

    def transform_node(
        self, new_node: Any, old_node: Any, n_samples_node: Any, val_map: Dict[Any, Any]
    ) -> Any:
        """
        Applies a transformation to a node. Default implementation copies the node.

        Args:
            new_node (Any): The new node being transformed.
            old_node (Any): The original node in the graph.
            n_samples_node (Any): The node representing 'n_samples'.
            val_map (Dict[Any, Any]): A mapping of original nodes to new nodes.

        Returns:
            Any: The transformed node.
        """
        return new_node

    def transform_forward(self, module: Any) -> Any:
        """
        Transforms the forward pass of a module. Default implementation returns the module unchanged.

        Args:
            module (Any): The module to transform.

        Returns:
            Any: The transformed module.
        """
        return module

    def add_methods(self, module: Any) -> Any:
        """
        Adds methods to the module.

        Args:
            module (Any): The module to add methods to.

        Returns:
            Any: The module with added methods.
        """
        return add_functions_to_module(module, self.methods)


class GaussianTransformation(Transformation):
    def __init__(self, custom_methods: Optional[Dict[str, Callable]] = None) -> None:
        """
        Transformation for Gaussian parameters, focusing on sampling.

        Args:
            custom_methods (Optional[Dict[str, Callable]]): Additional user-defined methods.
        """
        base_methods = {"kl_divergence": kl_divergence}
        super().__init__(base_methods, custom_methods)

    def transform_parameter(self, param: Any) -> Any:
        return to_gaussian(param)

    def transform_node(
        self, new_node: Any, old_node: Any, n_samples_node: Any, val_map: Dict[Any, Any]
    ) -> Any:
        return random_sample_node(new_node, old_node, n_samples_node, val_map)

    def transform_forward(self, module: Any) -> Any:
        return vmap_forward_transformer(module)


class GaussianDeterministicTransformation(GaussianTransformation):
    def transform_node(
        self, new_node: Any, old_node: Any, n_samples_node: Any, val_map: Dict[Any, Any]
    ) -> Any:
        return gaussian_mean_node(new_node, old_node, n_samples_node, val_map)


class ParticleTransformation(Transformation):
    def __init__(self, custom_methods: Optional[Dict[str, Callable]] = None) -> None:
        """
        Transformation for particle-based parameters.

        Args:
            custom_methods (Optional[Dict[str, Callable]]): Additional user-defined methods.
        """
        base_methods = {
            "named_particles": named_particles,
            "all_particles": all_particles,
            "compute_kernel_matrix": compute_kernel_matrix,
            "perturb_gradients": perturb_gradients,
        }
        super().__init__(base_methods, custom_methods)

    def transform_parameter(self, param: Any) -> Any:
        return to_particle(param)

    def transform_node(
        self, new_node: Any, old_node: Any, n_samples_node: Any, val_map: Dict[Any, Any]
    ) -> Any:
        return random_sample_node(new_node, old_node, n_samples_node, val_map)

    def transform_forward(self, module: Any) -> Any:
        return vmap_forward_transformer(module)
