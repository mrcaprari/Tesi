from typing import Any, Callable, Dict, Optional

import torch
import torch.fx
from torch.fx import GraphModule


class VmapTransformer(torch.fx.Transformer):
    """
    A custom FX transformer designed to modify the computation graph of a
    `torch.fx.GraphModule` to enable vectorized computations using `torch.vmap`.

    This transformer wraps functions in the forward computation graph with
    `torch.vmap`, allowing efficient batched operations while maintaining the
    original computation logic.
    """

    def __init__(self, module: GraphModule) -> None:
        """
        Initialize the VmapTransformer.

        Args:
            module (GraphModule): The computation graph module to transform.
        """
        super().__init__(module)
        self.in_dims: Optional[Any] = None  # Specifies input dimensions for vmap.

    def vmap_wrapper(self, target: Callable, in_dims: Optional[Any]) -> Callable:
        """
        Wrap a callable with a vmap for vectorized computations.

        Args:
            target (Callable): The target function to wrap.
            in_dims (Optional[Any]): Specifies which inputs to vectorize over.

        Returns:
            Callable: A wrapped function that applies vectorized computation.
        """

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            return torch.vmap(target, in_dims=in_dims)(*args, **kwargs)

        return wrapped

    def run_node(self, node: torch.fx.Node) -> Any:
        """
        Process a node in the computation graph.

        If the node represents a function call, determine which inputs are
        eligible for vectorization and store the corresponding input dimensions.

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
        Override the behavior of function calls in the computation graph.

        If input dimensions (`in_dims`) are specified, wrap the target function
        with `vmap` to enable vectorized computation. Otherwise, call the function
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
