from typing import Any, Dict, List, Optional, Union

import torch
import torch.fx


class PreparatoryTracer(torch.fx.Tracer):
    def __init__(
        self, transform_list: Optional[List[str]] = None, *args: Any, **kwargs: Any
    ) -> None:
        """
        Initializes the PreparatoryTracer with an optional list of transformation prefixes.

        Args:
            transform_list (Optional[List[str]]): A list of prefixes to determine which attributes to transform.
            *args (Any): Positional arguments for the parent class.
            **kwargs (Any): Keyword arguments for the parent class.
        """
        super().__init__(*args, **kwargs)
        self.transform_list = transform_list

    def is_leaf_module(
        self, module: torch.nn.Module, module_qualified_name: str
    ) -> bool:
        """
        Treat all modules as non-leaf modules, enabling full tracing of their logic.

        In FX tracing, a "leaf module" is treated as a single atomic node, and its internal
        operations are not traced. By overriding this behavior to treat all modules as non-leaf,
        we ensure that all function calls inside the module are captured in the FX graph.

        This allows to perform detailed graph-level transformations.

        Args:
            module (torch.nn.Module): The module being checked.
            module_qualified_name (str): The qualified name of the module.

        Returns:
            bool: Always returns False, treating all modules as non-leaf.
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
        Creates a node in the FX graph and marks it for transformation based on its operation and metadata.

        Args:
            kind (str): The kind of node (e.g., 'placeholder', 'call_function').
            target (Union[str, torch.nn.Module]): The target of the node.
            args (Any): Positional arguments for the node.
            kwargs (Dict[str, Any]): Keyword arguments for the node.
            name (Optional[str]): Optional name for the node.
            type_expr (Optional[Any]): Optional type expression for the node.

        Returns:
            torch.fx.Node: The created node with updated metadata.
        """
        node = super().create_node(kind, target, args, kwargs, name, type_expr)

        if node.op == "get_attr":
            node.meta["transform"] = self._should_transform_get_attr(node.target)

        if node.op == "call_function" and self._has_transformable_args(node.args):
            node.meta["transform"] = True

        if node.op == "call_method" and self._has_transformable_args(node.args):
            self._convert_call_method_to_function(node, target, args)

        return node

    def _should_transform_get_attr(self, target: str) -> bool:
        """
        Determines if a 'get_attr' node should be marked for transformation.

        Args:
            target (str): The target attribute name.

        Returns:
            bool: True if the attribute should be transformed, False otherwise.
        """
        if not self.transform_list:
            return True
        return any(target.startswith(transform) for transform in self.transform_list)

    def _has_transformable_args(self, args: Any) -> bool:
        """
        Checks if any argument in a node's arguments is marked for transformation.

        Args:
            args (Any): The arguments to check.

        Returns:
            bool: True if any argument has 'transform' metadata set to True, False otherwise.
        """
        return any(getattr(arg, "meta", {}).get("transform", False) for arg in args)

    def _convert_call_method_to_function(
        self, node: torch.fx.Node, target: Union[str, torch.nn.Module], args: Any
    ) -> None:
        """
        Converts a 'call_method' node into a 'call_function' node using a functional equivalent, if available.

        Args:
            node (torch.fx.Node): The node to modify.
            target (Union[str, torch.nn.Module]): The method target.
            args (Any): The arguments for the method.
        """
        func = getattr(torch, target, None) or getattr(
            torch.nn.functional, target, None
        )
        if func is not None:
            node.op = "call_function"
            node.target = func
            node.args = (args[0], *args[1:])
            node.meta["transform"] = True
