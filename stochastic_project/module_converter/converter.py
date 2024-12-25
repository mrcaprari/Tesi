import torch
import torch.fx
import torch.nn as nn

from stochastic_project.module_converter import PreparatoryTracer
from stochastic_project.transformations import BaseTransformation

class ModuleConverter:
    def __init__(self, transformation_logic):
        self.tracer = PreparatoryTracer
        # self.base_transformer = BaseTransformation
        self.transformation_logic = transformation_logic
        
    def last_placeholder(self, graph):
        for node in reversed(graph.nodes):
            if node.op == 'placeholder':
                return node
        return None

    def preliminary_graph(self, module, parameter_list):
        return self.tracer(parameter_list).trace(module)

    def transform_graph(self, module, parameter_list):
        preliminary_graph = self.preliminary_graph(module, parameter_list)
        last_placeholder = self.last_placeholder(preliminary_graph)

        new_graph = self._create_transformed_graph(
            preliminary_graph,
            last_placeholder,
        )

        return new_graph

    def _create_transformed_graph(self, preliminary_graph, last_placeholder):
        with preliminary_graph.inserting_after(last_placeholder):
            n_samples_node = preliminary_graph.placeholder(name="n_samples")
            new_graph = torch.fx.Graph()
            val_map = {}

            for node in preliminary_graph.nodes:
                new_node = new_graph.node_copy(node, lambda n: val_map[n])
                
                if node.meta.get('transform', False):
                    new_node = self.transformation_logic.transform_node(new_node, node, n_samples_node, val_map)

                val_map[node] = new_node
            return new_graph

    def transform_module(self, module, parameter_list):
        new_module = module
        preliminary_graph = self.preliminary_graph(module, parameter_list)
        for node in preliminary_graph.nodes:
            if node.meta.get('transform', False):
                if node.op == "get_attr":
                    self._transform_parameter(new_module, node.target)
        return new_module

    def _transform_parameter(self, module, name):
            attrs = name.split('.')
            *path, param_name = attrs
            submodule = module
            for attr in path:
                submodule = getattr(submodule, attr)
            param = getattr(submodule, param_name)            
            delattr(submodule, param_name)
            new_param = self.transformation_logic.transform_parameter(param)
            submodule.register_module(param_name, new_param)
            return True

    def convert(self, module, parameter_list):
        new_graph = self.transform_graph(module, parameter_list)
        new_module = self.transform_module(module, parameter_list)
        transformed_module = torch.fx.GraphModule(new_module, new_graph)
        return self.transformation_logic.transform_forward(transformed_module)
