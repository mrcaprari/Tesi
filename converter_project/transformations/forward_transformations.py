from typing import Any, Callable, Dict, Optional

import torch
import torch.fx
from torch.fx import GraphModule


def vmap_forward_transformer(module: GraphModule) -> GraphModule:
    """
    Applies `torch.vmap` to the forward pass of a module, transforming its function calls
    to be compatible with batched inputs.

    This transformer modifies `call_function` nodes in the FX graph to use `torch.vmap`
    with the specified `in_dims` metadata.

    Args:
        module (GraphModule): The module whose forward pass should be transformed.

    Returns:
        GraphModule: A transformed module with `torch.vmap` applied to relevant function calls.
    """

    def vmap_wrapper(target: Callable, in_dims: Optional[Any]) -> Callable:
        """
        Wraps a function with `torch.vmap`, setting the specified `in_dims`.

        Args:
            target (Callable): The function to wrap with `torch.vmap`.
            in_dims (Optional[Any]): The input dimensions to use for `torch.vmap`.

        Returns:
            Callable: The wrapped function with `torch.vmap` applied.
        """

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            return torch.vmap(target, in_dims=in_dims)(*args, **kwargs)

        return wrapped

    class ForwardTransformer(torch.fx.Transformer):
        """
        An FX graph transformer that applies `torch.vmap` to function calls during execution.

        Attributes:
            in_dims (Optional[Any]): The input dimensions for `torch.vmap` in the current context.
        """

        def __init__(self, module: GraphModule) -> None:
            super().__init__(module)
            self.in_dims: Optional[Any] = None

        def run_node(self, node: torch.fx.Node) -> Any:
            """
            Processes each node in the graph during transformation.

            Updates the `in_dims` attribute based on the node's metadata.

            Args:
                node (torch.fx.Node): The node to process.

            Returns:
                Any: The result of executing the node.
            """
            if node.op == "call_function":
                self.in_dims = node.meta.get("in_dims", None)
            return super().run_node(node)

        def call_function(
            self, target: Callable, args: Any, kwargs: Dict[str, Any]
        ) -> Any:
            """
            Transforms `call_function` nodes by wrapping their target in `torch.vmap`.

            Args:
                target (Callable): The original function being called.
                args (Any): Positional arguments for the function.
                kwargs (Dict[str, Any]): Keyword arguments for the function.

            Returns:
                Any: The result of the transformed function call.
            """
            if self.in_dims is not None:
                vmap_fn = vmap_wrapper(target, self.in_dims)
                return super().call_function(vmap_fn, args, kwargs)
            return super().call_function(target, args, kwargs)

    return ForwardTransformer(module).transform()
