import copy
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch
from torch.fx import Graph, Node

from ..distributions import (
    DistributionFactory,
    Gaussian,
    Particle,
    Prior,
    RandomParameter,
)


class Adapter(ABC):
    """
    Base class for adapting parameters and their corresponding nodes in a torch.fx computation graph.

    Attributes:
        prepared_graph (Graph): The original graph to be transformed.
        last_placeholder (Optional[Node]): The last placeholder node in the graph, used for insertion points.
    """

    def __init__(self, prepared_graph: Graph) -> None:
        """
        Initializes the Adapter with a prepared graph.

        Args:
            prepared_graph (Graph): The graph that will be transformed by the adapter.
        """
        self.prepared_graph = prepared_graph
        self.last_placeholder = self._find_last_placeholder()

    @abstractmethod
    def adapt_node(
        self,
        new_node: Node,
        old_node: Node,
        n_samples_node: Node,
        val_map: Dict[Node, Node],
    ) -> Node:
        """
        Adapts a node during graph transformation. To be overridden by subclasses.

        Args:
            new_node (Node): The newly created node in the transformed graph.
            old_node (Node): The corresponding node in the original graph.
            n_samples_node (Node): A placeholder node representing the number of samples.
            val_map (Dict[Node, Node]): A mapping of original nodes to new nodes.

        Returns:
            Node: The adapted node.
        """
        pass

    @abstractmethod
    def adapt_parameter(self, parameter: torch.nn.Parameter) -> torch.nn.Module:
        """
        Adapts a parameter into a new module. This method must be implemented by subclasses.

        Args:
            parameter (torch.nn.Parameter): The parameter to adapt.

        Returns:
            torch.nn.Module: The adapted parameter as a module.
        """
        pass

    def adapt_module(self, module: torch.nn.Module) -> tuple[torch.nn.Module, Graph]:
        """
        Constructs a new graph module by adapting the prepared graph and its parameters.

        Args:
            module (torch.nn.Module): The original module to be transformed.

        Returns:
            tuple[torch.nn.Module, Graph]: The transformed module and its computation graph.
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
                    self._adapt_parameter(new_module, node.target)

            val_map[node] = new_node

        return new_module, new_graph

    def _adapt_parameter(self, module: torch.nn.Module, name: str) -> bool:
        """
        Adapts a parameter by replacing it with a transformed version in the module.

        Args:
            module (torch.nn.Module): The module containing the parameter.
            name (str): The name of the parameter to adapt.

        Returns:
            bool: True if the parameter was successfully adapted.
        """
        attrs = name.split(".")
        *path, param_name = attrs
        submodule = module

        for attr in path:
            submodule = getattr(submodule, attr)

        param = getattr(submodule, param_name)
        delattr(submodule, param_name)

        new_param = self.adapt_parameter(param)
        submodule.register_module(param_name, new_param)

        return True

    def _find_last_placeholder(self) -> Optional[Node]:
        """
        Finds the last placeholder node in the prepared graph.

        Returns:
            Optional[Node]: The last placeholder node, or None if not found.
        """
        return next(
            (
                node
                for node in reversed(self.prepared_graph.nodes)
                if node.op == "placeholder"
            ),
            None,
        )


class GaussianAdapter(Adapter):
    """
    Adapter for transforming parameters and nodes using a Gaussian-based strategy.

    The GaussianAdapter specializes in adapting parameters by associating them with a
    Gaussian distribution and transforming graph nodes to reflect this new representation.

    Methods:
        adapt_node: Modifies a graph node to incorporate sampling logic for a Gaussian
                    parameter. Specifically, it replaces `get_attr` operations with
                    `call_module` operations that handle sampling.
        adapt_parameter: Replaces a parameter with a random variable following a
                         Gaussian distribution. The random variable includes a prior
                         distribution and a small standard deviation.

    """

    def adapt_node(
        self,
        new_node: Node,
        old_node: Node,
        n_samples_node: Node,
        val_map: Dict[Node, Node],
    ) -> Node:
        if old_node.op == "get_attr":
            new_node.name = f"sampled_{old_node.name}"
            new_node.op = "call_module"
            new_node.args = (val_map[n_samples_node],)
        return new_node

    def adapt_parameter(self, parameter: torch.nn.Parameter) -> torch.nn.Module:
        param_prior = DistributionFactory.create(Gaussian, Prior, shape=parameter.shape)
        random_param = DistributionFactory.create(
            Gaussian,
            RandomParameter,
            mu=parameter,
            prior=param_prior,
            std=torch.tensor(0.01),
        )
        return random_param


class ParticleAdapter(Adapter):
    """
    Adapter for transforming parameters and nodes using a Particle-based strategy.

    The ParticleAdapter replaces parameters with a particle-based representation,
    where particles approximate a target distributio.

    Methods:
        adapt_node: Modifies a graph node to support sampling of a particle-based parameter.
                    It replaces `get_attr` operations with `call_module` operations that
                    handle particle-based computations.
        adapt_parameter: Replaces a parameter with a particle-based representation,
                         which approximates the parameter's prior distribution using particles.

    Example Use Case:
        - Particle filters for state estimation in dynamic systems.
        - Probabilistic models leveraging particle approximations for uncertainty representation.
    """

    def adapt_node(
        self,
        new_node: Node,
        old_node: Node,
        n_samples_node: Node,
        val_map: Dict[Node, Node],
    ) -> Node:
        if old_node.op == "get_attr":
            new_node.name = f"sampled_{old_node.name}"
            new_node.op = "call_module"
            new_node.args = (val_map[n_samples_node],)
        return new_node

    def adapt_parameter(self, parameter: torch.nn.Parameter) -> torch.nn.Module:
        param_prior = DistributionFactory.create(Gaussian, Prior, shape=parameter.shape)
        particle_param = Particle(param_prior)
        return particle_param
