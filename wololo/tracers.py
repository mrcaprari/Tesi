from typing import Any, Dict, List, Optional, Union

import torch
import torch.fx


class PreparatoryTracer(torch.fx.Tracer):
    def __init__(
        self, transform_list: Optional[List[str]] = None, *args: Any, **kwargs: Any
    ) -> None:
        """
        A custom tracer for preparing computation graphs for transformations.

        This tracer overrides the default behavior of `torch.fx.Tracer` to ensure
        that no module is treated as a leaf module, enabling fine-grained control during
        subsequent transformation steps.
        It also adds "transform" metadata to nodes corresponding to parameters in the transform_list,
        or to nodes corresponding to functions having arguments in the transform_list.

        Attributes:
            transform_list (Optional[List[str]]): The list of parameters to be transformed.
                If None, all parameters will be transformed.
        """

        super().__init__(*args, **kwargs)
        self.transform_list = transform_list

    def is_leaf_module(
        self, module: torch.nn.Module, module_qualified_name: str
    ) -> bool:
        """
        Override to ensure no module is treated as a leaf module, enabling fine-grained control during
        subsequent transformation steps.

        Args:
            module (torch.nn.Module): The module being traced.
            module_qualified_name (str): Qualified name of the module.

        Returns:
            bool: Always returns False to prevent modules being treated as leaf nodes.
        """
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
        """
        Creates a new node in the computation graph and annotates it with "transform" metadata.
        This extends the base `create_node` functionality by adding "transform" metadata
        to each node. Nodes of type "get_attr" are marked based on whether their target should
        be transformed, while other nodes are marked if any of their arguments are marked
        for transformation.

        Args:
            kind (str): The type of the node (e.g., call_function, call_module).
            target (Union[str, torch.nn.Module]): The target of the node.
            args (Any): Positional arguments for the node.
            kwargs (Dict[str, Any]): Keyword arguments for the node.
            name (Optional[str]): Optional name for the node.
            type_expr (Optional[Any]): Optional type expression for the node.

        Returns:
            torch.fx.Node: The newly created node with the appropriate metadata.
        """
        node = super().create_node(kind, target, args, kwargs, name, type_expr)

        if node.op == "get_attr":
            node.meta["transform"] = self._should_transform_get_attr(node.target)
        else:
            node.meta["transform"] = self._has_transformable_args(node.args)
        return node

    def _should_transform_get_attr(self, target: str) -> bool:
        """
        Determines if a 'get_attr' node should be transformed.

        A node should be transformed if it matches one of the prefixes in
        the `transform_list`. If `transform_list` is None, all targets are
        considered transformable.

        Args:
            target (str): The target attribute name.

        Returns:
            bool: True if the target should be transformed; otherwise, False.
        """
        if not self.transform_list:
            return True
        return any(target.startswith(transform) for transform in self.transform_list)

    def _has_transformable_args(self, args: Any) -> bool:
        """
        Examines the "transform" metadata of the arguments to determine
        if the node should be marked for transformation.

        Args:
            args (Any): The current node arguments to check.

        Returns:
            bool: True if any argument is marked for transformation, False otherwise.
        """
        return any(getattr(arg, "meta", {}).get("transform", False) for arg in args)
