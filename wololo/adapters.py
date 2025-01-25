import copy
from typing import Any, Dict, Optional

import torch
from torch.fx import Graph, Node


class Adapter:
    """
    A class for substituting parameters with given stochastic parameters and updates the computation graph
    structure to incorporate these changes.

    Attributes:
        prepared_graph (Graph): The graph of the deterministic module to be transformed, created by a torch.fx.Tracer.
        last_placeholder (Optional[Node]): The last placeholder node in the graph, used as the insertion point
            for the new placeholder node "n_samples".
    """

    def __init__(self, prepared_graph: Graph) -> None:
        """
        Initializes the Adapter with the prepared computation graph.

        Args:
            prepared_graph (Graph): The computation graph to be adapted created by a `torch.fx.Tracer`.
        """
        self.prepared_graph = prepared_graph
        self.last_placeholder = self._find_last_placeholder()

    def adapt_node(
        self,
        new_node: Node,
        old_node: Node,
        n_samples_node: Node,
        val_map: Dict[Node, Node],
    ) -> Node:
        """
        Transforms nodes with "transform=True" metadata by substituting "get_attr"
        operations with "call_module". This enables the substitution with stochastic
        modules that realize random parameters.

        Args:
            new_node (Node): The node to transform.
            old_node (Node): The original node in the graph.
            n_samples_node (Node): A placeholder for sample count.
            val_map (Dict[Node, Node]): A mapping of old nodes to their new equivalents.

        Returns:
            Node: The transformed node.
        """
        if old_node.op == "get_attr":
            new_node.name = f"sampled_{old_node.name}"
            new_node.op = "call_module"
            new_node.args = (val_map[n_samples_node],)
        return new_node

    def adapt_parameter(
        self,
        parameter: torch.nn.Parameter,
        stochastic_module_cls: type[torch.nn.Module],
        **kwargs,  # Additional arguments for the stochastic module
    ) -> torch.nn.Module:
        """
        Replaces a parameter with an instance of torch.nn.Module responsible for
        generating realizations of the random parameter in its `forward()` method.

        Args:
            parameter (torch.nn.Parameter): The parameter to replace.
            stochastic_module_cls (type[torch.nn.Module]): A `torch.nn.Module` class for
                stochastic sampling, such as random parameter realizations.

        Returns:
            torch.nn.Module: A stochastic module initialized with the parameter.
        """

        if not issubclass(stochastic_module_cls, torch.nn.Module):
            raise TypeError(
                f"{stochastic_module_cls} must be a subclass of torch.nn.Module."
            )
        return stochastic_module_cls(parameter, **kwargs)

    def adapt_module(
        self,
        module: torch.nn.Module,
        stochastic_module_cls: type[torch.nn.Module],
        **kwargs,  # Additional arguments for the stochastic module
    ) -> tuple[torch.nn.Module, Graph]:
        """
        Adapts the module and its computation graph by replacing parameters with stochastic
        modules and transforming nodes previously marked for transformation.

        Args:
            module (torch.nn.Module): The module to adapt.
            stochastic_module_cls (type[torch.nn.Module]): A `torch.nn.Module` class for stochastic parameters.

        Returns:
            tuple[torch.nn.Module, Graph]: The adapted module and the transformed graph.
        """
        new_module = copy.deepcopy(module)
        with self.prepared_graph.inserting_after(self.last_placeholder):
            n_samples_node = self.prepared_graph.placeholder(name="n_samples")

        new_graph = Graph()
        val_map: Dict[Node, Node] = {}

        for node in self.prepared_graph.nodes:
            new_node = new_graph.node_copy(node, lambda n: val_map[n])
            if node.meta.get("transform", False):
                new_node = self.adapt_node(new_node, node, n_samples_node, val_map)
                if node.op == "get_attr":
                    self._adapt_parameter(
                        new_module, stochastic_module_cls, node.target, **kwargs
                    )

            val_map[node] = new_node

        return new_module, new_graph

    def _adapt_parameter(
        self,
        module: torch.nn.Module,
        stochastic_module_cls: type[torch.nn.Module],
        name: str,
        **kwargs,  # Additional arguments for the stochastic module
    ) -> bool:
        """
        Replaces a parameter in the module with a torch.nn.Module encoding the
        stochastic parameter logic.

        Args:
            module (torch.nn.Module): The module containing the parameter.
            name (str): The fully qualified name of the parameter to replace.

        Returns:
            bool: True if the parameter was successfully replaced.
        """
        attrs = name.split(".")
        *path, param_name = attrs
        submodule = module

        for attr in path:
            submodule = getattr(submodule, attr)

        param = getattr(submodule, param_name)
        delattr(submodule, param_name)

        new_param = self.adapt_parameter(param, stochastic_module_cls, **kwargs)
        submodule.register_module(param_name, new_param)

        return True

    def _find_last_placeholder(self) -> Optional[Node]:
        """
        Finds the last placeholder node in the graph, used as an insertion point
        for the "n_samples" placeholder node.

        Returns:
            Optional[Node]: The last placeholder node or None if not found.
        """
        return next(
            (
                node
                for node in reversed(self.prepared_graph.nodes)
                if node.op == "placeholder"
            ),
            None,
        )
