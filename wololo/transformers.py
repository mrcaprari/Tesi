from typing import Any, Callable, Dict, Optional

import torch
import torch.fx
from torch.fx import GraphModule


class VmapTransformer(torch.fx.Transformer):
    """
    A custom FX transformer for enabling batched computations using `torch.vmap`.

    This transformer modifies the computation graph of a `torch.fx.GraphModule`,
    automatically wrapping functions with `torch.vmap` to efficiently handle batched
    parameter dimensions while preserving the original computation logic.
    """

    def __init__(self, module: GraphModule) -> None:
        """
        Initialize the VmapTransformer with the given computation graph module.

        Args:
            module (GraphModule): The computation graph module to transform.
        """
        super().__init__(module)
        self.in_dims: Optional[Any] = None  # Specifies input dimensions for vmap.

    def vmap_wrapper(self, target: Callable, in_dims: Optional[Any]) -> Callable:
        """
        Wraps a callable with `torch.vmap` for vectorized computations.

        Args:
            target (Callable): The function to wrap with vectorization.
            in_dims (Optional[Any]): Dimensions of inputs to vectorize over.

        Returns:
            Callable: A function wrapped with `torch.vmap` for batched computations.
        """

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            return torch.vmap(target, in_dims=in_dims)(*args, **kwargs)

        return wrapped

    def run_node(self, node: torch.fx.Node) -> Any:
        """
        Processes a node in the computation graph during the transformation process.

        For function call nodes, this method determines which input dimensions need
        vectorization and updates the 'in_dims' attribute accordingly.

        Args:
            node (torch.fx.Node): The node to process.

        Returns:
            Any: The result of running the node.
        """
        if node.op == "call_function":
            current_in_dims = tuple(
                0 if arg.meta.get("transform", False) else None for arg in node.args
            )
            self.in_dims = current_in_dims
        return super().run_node(node)

    def call_function(self, target: Callable, args: Any, kwargs: Dict[str, Any]) -> Any:
        """
        Overrides the behavior of function calls in the computation graph.

        If input dimensions (`in_dims`) are specified, wrap the target function
        with `torch.vmap` to enable vectorized computation. Otherwise, call the function
        as usual.

        Args:
            target (Callable): The target function to call.
            args (Any): Positional arguments for the function call.
            kwargs (Dict[str, Any]): Keyword arguments for the function call.

        Returns:
            Any: The result of the function call.
        """
        if self.in_dims is not None:
            vmap_fn = self.vmap_wrapper(target, self.in_dims)
            return super().call_function(vmap_fn, args, kwargs)
        return super().call_function(target, args, kwargs)
