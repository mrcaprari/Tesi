import torch
import torch.fx
from . import custom_tracers
from . import utils

class GraphFactory():
    def __init__(self, original_module: torch.nn.Module, tracer: torch.fx.Tracer):
        self.original_graph = tracer.trace(original_module)
        self.stochastic_nodes = set(
            node for node in self.original_graph.nodes
            if node.op == "get_attr" and utils.points_to_stochastic_parameters(original_module, node.target)
        )
        if not self.stochastic_nodes:
            raise ValueError("No stochastic nodes were found in the traced graph. "
            "This GraphFactory requires at least one stochastic node to be useful.")

    def create_deterministic_graph(self) -> torch.fx.GraphModule:
        deterministic_graph = torch.fx.Graph()
        deterministic_node_map = {}

        for node in self.original_graph.nodes:
            if node in self.stochastic_nodes:
                # Create the first node pointing to the submodule
                submodule_node = deterministic_graph.create_node(
                    op="get_attr",
                    target=node.target,
                )

                # Create the second node accessing the "mu" parameter of the submodule
                mu_node = deterministic_graph.create_node(
                    op="get_attr",
                    target=f"{node.target}.mu",
                    args=(submodule_node,)
                )

                # Map the original node to the new "mu" node
                deterministic_node_map[node] = mu_node
            
            else:
                # Copy the node as is
                deterministic_node_map[node] = deterministic_graph.node_copy(node, 
                                                                            lambda n: deterministic_node_map[n])
            
        return deterministic_graph

    def create_BBVI_graph(self):
        BBVI_graph = torch.fx.Graph()
        BBVI__node_map = {}

        # # Iterate through the original graph and copy placeholders first
        for node in self.original_graph.nodes:
            if node.op == "placeholder":
                # Copy existing placeholders into stoch_graph
                new_placeholder = BBVI__graph.create_node(
                    op="placeholder",
                    target=node.target,
                    args=(),
                    kwargs={},
                    name=node.name
                )
                BBVI__node_map[node] = new_placeholder
                last_placeholder = new_placeholder

        with BBVI_graph.inserting_after(last_placeholder):
            n_samples_node = BBVI_graph.placeholder(name="n_samples", default_value=1)

        # Copy other nodes
        for node in self.original_graph.nodes:
            if node.op != "placeholder":
                if node in self.stochastic_nodes:
                    # Create the first node pointing to the submodule
                    submodule_node = BBVI_graph.create_node(
                        op="get_attr",
                        target=node.target,
                    )

                    # Create the second node accessing the "mu" parameter of the submodule
                    realization_node = BBVI_graph.create_node(
                        op="call_module",
                        target=f"{node.target}",
                        args=(n_samples_node,)
                    )

                    realization_node.meta = {**node.meta, 'additional_map_dim': True}

                    for user in utils.all_users(node):
                        user.meta = {**user.meta, 'additional_map_dim': True}

                    BBVI_node_map[user] = user
                    BBVI_node_map[node] = realization_node
                else:
                    new_node = BBVI_graph.node_copy(node,
                                                        lambda n: BBVI_node_map[n])
                    new_node.meta = {**node.meta}
                    BBVI_node_map[node] = new_node

        return BBVI_graph

