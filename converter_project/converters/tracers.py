from typing import Any, Dict, List, Optional, Union

import torch
import torch.fx


class PreparatoryTracer(torch.fx.Tracer):
    def __init__(
        self, transform_list: Optional[List[str]] = None, *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.transform_list = transform_list

    def is_leaf_module(
        self, module: torch.nn.Module, module_qualified_name: str
    ) -> bool:
        return False

    def create_node(
        self,
        kind: str,
        target: Union[str, torch.nn.Module],
        args: Any,
        kwargs: Dict[str, Any],
        name: Optional[str] = None,
        type_expr: Optional[Any] = None,
    ) -> torch.fx.Node:
        node = super().create_node(kind, target, args, kwargs, name, type_expr)

        if node.op == "get_attr":
            node.meta["transform"] = self._should_transform_get_attr(node.target)
        else:
            node.meta["transform"] = self._has_transformable_args(node.args)
        return node

    def _should_transform_get_attr(self, target: str) -> bool:
        if not self.transform_list:
            return True
        return any(target.startswith(transform) for transform in self.transform_list)

    def _has_transformable_args(self, args: Any) -> bool:
        return any(getattr(arg, "meta", {}).get("transform", False) for arg in args)
