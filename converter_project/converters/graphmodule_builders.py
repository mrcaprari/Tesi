import copy
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.fx
from torch.fx import Graph, Node

from converter_project.distributions import (
    DistributionFactory,
    Gaussian,
    Particle,
    Prior,
    RandomParameter,
)


class GraphModuleBuilder:
    def __init__(self, prepared_graph):
        self.prepared_graph = prepared_graph
        self.last_placeholder = self._find_last_placeholder()

    def transform_node(self, new_node, old_node, n_samples_node, val_map):
        return new_node

    def transform_parameter(self, parameter):
        return torch.nn.Linear(1, 1)

    def build_graph_module(self, module):
        new_module = copy.deepcopy(module)
        with self.prepared_graph.inserting_after(self.last_placeholder):
            n_samples_node = self.prepared_graph.placeholder(name="n_samples")

        new_graph = Graph()
        val_map: Dict[Node, Node] = {}

        for node in self.prepared_graph.nodes:
            new_node = new_graph.node_copy(node, lambda n: val_map[n])
            if node.meta.get("transform", False):
                new_node = self.transform_node(new_node, node, n_samples_node, val_map)
                if node.op == "get_attr":
                    self._transform_parameter(new_module, node.target)

            val_map[node] = new_node

        return new_module, new_graph

    def _transform_parameter(self, module: torch.nn.Module, name: str) -> bool:
        attrs = name.split(".")
        *path, param_name = attrs
        submodule = module

        for attr in path:
            submodule = getattr(submodule, attr)

        param = getattr(submodule, param_name)
        delattr(submodule, param_name)

        new_param = self.transform_parameter(param)
        submodule.register_module(param_name, new_param)

        return True

    def _find_last_placeholder(self) -> Optional[Node]:
        return next(
            (
                node
                for node in reversed(self.prepared_graph.nodes)
                if node.op == "placeholder"
            ),
            None,
        )


def get_meta(arg: Any, key: str, default: Optional[Any] = None) -> Any:
    return arg.meta.get(key, default) if hasattr(arg, "meta") else default


class GaussianGraphModuleBuilder(GraphModuleBuilder):
    def transform_node(self, new_node, old_node, n_samples_node, val_map):
        if old_node.op == "get_attr":
            new_node.name = f"sampled_{old_node.name}"
            new_node.op = "call_module"
            new_node.args = (val_map[n_samples_node],)
        return new_node

    def transform_parameter(self, parameter):
        param_prior = DistributionFactory.create(Gaussian, Prior, shape=parameter.shape)
        random_param = DistributionFactory.create(
            Gaussian,
            RandomParameter,
            mu=parameter,
            prior=param_prior,
            std=torch.tensor(0.01),
        )
        return random_param


class ParticleGraphModuleBuilder(GraphModuleBuilder):
    def transform_node(self, new_node, old_node, n_samples_node, val_map):
        if old_node.op == "get_attr":
            new_node.name = f"sampled_{old_node.name}"
            new_node.op = "call_module"
            new_node.args = (val_map[n_samples_node],)
        return new_node

    def transform_parameter(self, parameter):
        param_prior = DistributionFactory.create(Gaussian, Prior, shape=parameter.shape)
        particle_param = Particle(param_prior)
        return particle_param
