import copy
from typing import Any, Dict, List, Optional

import torch
import torch.fx
from torch.fx import Graph, GraphModule, Node

from converter_project.converters.preparatory_tracer import PreparatoryTracer
from converter_project.transformations.transformations import Transformation


class ModuleConverter:
    def __init__(self, transformation_logic: Transformation) -> None:
        """
        Initializes the ModuleConverter with the given transformation logic.

        Args:
            transformation_logic (Any): An object encapsulating transformation logic for nodes and parameters.
        """
        self.tracer = PreparatoryTracer
        self.transformation_logic = transformation_logic

    def convert(
        self, module: torch.nn.Module, parameter_list: Optional[List[Any]] = None
    ) -> GraphModule:
        """
        Converts a module into a transformed module with updated graph and methods.

        Args:
            module (torch.nn.Module): The module to convert.
            parameter_list (Optional[List[Any]]): The parameters to use for conversion.

        Returns:
            GraphModule: The transformed module.
        """
        if parameter_list is None:
            parameter_list = []

        new_graph = self._transform_graph(module, parameter_list)
        new_module = self._transform_module(module, parameter_list)

        transformed_module = GraphModule(new_module, new_graph)
        transformed_module = self.transformation_logic.transform_forward(
            transformed_module
        )
        transformed_module = self.transformation_logic.add_methods(transformed_module)

        return transformed_module

    def _transform_graph(
        self, module: torch.nn.Module, parameter_list: List[Any]
    ) -> Graph:
        """
        Transforms the graph of the given module based on transformation logic.

        Args:
            module (torch.nn.Module): The module to transform.
            parameter_list (List[Any]): The parameters to transform with.

        Returns:
            Graph: The transformed graph.
        """
        preliminary_graph = self._preliminary_graph(module, parameter_list)
        last_placeholder = self._last_placeholder(preliminary_graph)
        return self._create_transformed_graph(preliminary_graph, last_placeholder)

    def _transform_module(
        self, module: torch.nn.Module, parameter_list: List[Any]
    ) -> torch.nn.Module:
        """
        Transforms the module by modifying its parameters based on transformation logic.


        Args:
            module (torch.nn.Module): The module to transform.
            parameter_list (List[Any]): The parameters to transform with.

        Returns:
            torch.nn.Module: The transformed module.
        """
        new_module = copy.deepcopy(module)
        preliminary_graph = self._preliminary_graph(module, parameter_list)

        for node in preliminary_graph.nodes:
            if node.meta.get("transform", False) and node.op == "get_attr":
                self._transform_parameter(new_module, node.target)

        return new_module

    def _create_transformed_graph(
        self, preliminary_graph: Graph, last_placeholder: Optional[Node]
    ) -> Graph:
        """
        Creates a new graph by applying transformation logic to nodes of the preliminary graph.

        Args:
            preliminary_graph (Graph): The original graph.
            last_placeholder (Optional[Node]): The last placeholder node in the graph.

        Returns:
            Graph: The transformed graph.
        """
        with preliminary_graph.inserting_after(last_placeholder):
            n_samples_node = preliminary_graph.placeholder(name="n_samples")
            new_graph = Graph()
            val_map: Dict[Node, Node] = {}

            for node in preliminary_graph.nodes:
                new_node = new_graph.node_copy(node, lambda n: val_map[n])

                if node.meta.get("transform", False):
                    new_node = self.transformation_logic.transform_node(
                        new_node, node, n_samples_node, val_map
                    )

                val_map[node] = new_node

            return new_graph

    def _transform_parameter(self, module: torch.nn.Module, name: str) -> bool:
        """
        Transforms a parameter of the module by applying the transformation logic.

        Args:
            module (torch.nn.Module): The module containing the parameter.
            name (str): The name of the parameter.

        Returns:
            bool: True if the transformation was successful.
        """
        attrs = name.split(".")
        *path, param_name = attrs
        submodule = module

        for attr in path:
            submodule = getattr(submodule, attr)

        param = getattr(submodule, param_name)
        delattr(submodule, param_name)

        new_param = self.transformation_logic.transform_parameter(param)
        submodule.register_module(param_name, new_param)

        return True

    def _last_placeholder(self, graph: Graph) -> Optional[Node]:
        """
        Finds the last placeholder node in the given graph.

        Args:
            graph (Graph): The graph to search.

        Returns:
            Optional[Node]: The last placeholder node if found, else None.
        """
        return next(
            (node for node in reversed(graph.nodes) if node.op == "placeholder"), None
        )

    def _preliminary_graph(
        self, module: torch.nn.Module, parameter_list: List[str]
    ) -> Graph:
        """
        Creates a preliminary graph by tracing the given module.

        Args:
            module (torch.nn.Module): The module to trace.
            parameter_list (List[str]): The list of fully qualified parameters to be transformed

        Returns:
            Graph: The traced graph.
        """
        return self.tracer(parameter_list).trace(module)
