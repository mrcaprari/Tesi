import torch
import torch.fx
from . import custom_tracers
from . import utils

def find_last_placeholder(graph: torch.fx.Graph):
    # Iterate over the nodes in reverse order
    for node in reversed(graph.nodes):
        if node.op == 'placeholder':
            return node
    return None  # No placeholder found

### Create Deterministic Graph
# with graph.inserting_after(graph._root):
#     new_node = graph.create_node(
#         op='placeholder',
#         target='n_samples',
#         args=(),
#         kwargs={},
#         name='n_samples'
#     )

# # initiating 'additional_map_dim' metadata
# for node in graph.nodes:
#     node.meta = {**node.meta, 'additional_map_dim': False}


def create_deterministic_graph(original_module: torch.nn.Module, 
                               tracer: torch.fx.Tracer = custom_tracers.NoLeafTracer()) -> torch.fx.GraphModule:
  
    original_graph = tracer.trace(original_module)
    utils.convert_to_stochastic_parameters(original_module)

    deterministic_graph = torch.fx.Graph()
    deterministic_node_map = {}

    for node in original_graph.nodes:
        if node.op == "get_attr" and utils.points_to_stochastic_parameters(original_module, node.target):
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
        
    return torch.fx.GraphModule(original_module, deterministic_graph)


### Stochastic Graph
def create_stochastic_graph(original_module: torch.nn.Module, 
                            tracer: torch.fx.Tracer = custom_tracers.NoLeafTracer()):
  
    original_graph = tracer.trace(original_module)
    utils.convert_to_stochastic_parameters(original_module)
    #graph_module = torch.fx.GraphModule(original_module, original_graph)  
    
    stochastic_graph = torch.fx.Graph()
    stochastic_node_map = {}

    # # Iterate through the original graph and copy placeholders first
    for node in original_graph.nodes:
        if node.op == "placeholder":
            # Copy existing placeholders into stoch_graph
            new_placeholder = stochastic_graph.create_node(
                op="placeholder",
                target=node.target,
                args=(),
                kwargs={},
                name=node.name
            )
            stochastic_node_map[node] = new_placeholder
            last_placeholder = new_placeholder

    with stochastic_graph.inserting_after(last_placeholder):
        n_samples_node = stochastic_graph.placeholder(name="n_samples", default_value=1)

    # Copy other nodes
    for node in original_graph.nodes:
        if node.op != "placeholder":
            if node.op == "get_attr" and utils.points_to_stochastic_parameters(original_module, node.target):
                # Create the first node pointing to the submodule
                submodule_node = stochastic_graph.create_node(
                    op="get_attr",
                    target=node.target,
                )

                # Create the second node accessing the "mu" parameter of the submodule
                realization_node = stochastic_graph.create_node(
                    op="call_module",
                    target=f"{node.target}",
                    args=(n_samples_node,)
                )

                realization_node.meta = {**node.meta, 'additional_map_dim': True}

                for user in utils.all_users(node):
                    user.meta = {**user.meta, 'additional_map_dim': True}

                stochastic_node_map[user] = user
                stochastic_node_map[node] = realization_node
            else:
                new_node = stochastic_graph.node_copy(node,
                                                      lambda n: stochastic_node_map[n])
                new_node.meta = {**node.meta}
                stochastic_node_map[node] = new_node

    # Return stochastic GraphModule
    return torch.fx.GraphModule(original_module, stochastic_graph)




