import torch
import torch.fx
import stochastic_parameter

def convert_to_stochastic_parameters(module: torch.nn.Module):
    for name, param in list(module.named_parameters()):  # Only transform parameters of the current module
        if param.requires_grad:
            # Replace with a StochasticParameter
            stochastic_param = stochastic_parameter.StochasticParameter.from_mu(param)
            delattr(module, name)  # Remove the old parameter
            module.register_module(name, stochastic_param)  # Add the new parameter

def points_to_stochastic_parameters(module: torch.nn.Module, node_target: str)-> bool:
    for attr in node_target.split('.'):
        module = getattr(module, attr)
    return isinstance(module, stochastic_parameter.StochasticParameter)

def all_users(node: torch.fx.Node) -> set:
    users = set()
    def collect_users(n):
        for user in n.users:
            if user not in users:
                users.add(user)
                collect_users(user)
    collect_users(node)
    return users

