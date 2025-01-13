from typing import Any, Callable, Dict, List, Optional

import torch
import torch.fx
from torch.fx import GraphModule


class VmapTransformer(torch.fx.Transformer):
    def __init__(self, module: GraphModule) -> None:
        super().__init__(module)
        self.in_dims: Optional[Any] = None

    def vmap_wrapper(self, target: Callable, in_dims: Optional[Any]) -> Callable:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            return torch.vmap(target, in_dims=in_dims)(*args, **kwargs)

        return wrapped

    def run_node(self, node: torch.fx.Node) -> Any:
        if node.op == "call_function":
            current_in_dims = tuple(
                0 if arg.meta.get("transform", False) else None for arg in node.args
            )
            self.in_dims = current_in_dims
        return super().run_node(node)

    def call_function(self, target: Callable, args: Any, kwargs: Dict[str, Any]) -> Any:
        if self.in_dims is not None:
            vmap_fn = self.vmap_wrapper(target, self.in_dims)
            return super().call_function(vmap_fn, args, kwargs)
        return super().call_function(target, args, kwargs)
