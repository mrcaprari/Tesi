import torch
import torch.fx
from torch.fx import GraphModule
from stochastic_project.stochastic_parameter import *
from typing import Iterator, Union, Dict, Any


# Register the wrapped function with FX
def vmap_builder(target_fn, current_in_dims):
    return torch.vmap(func=target_fn, 
                      in_dims=current_in_dims)


prior = GaussianPrior(shape=(10,5))
prova1 = Particle(prior=prior)
prova1(n_particles=3)
prova2 = GaussianParameter(prior.shape, prior = prior)


class NewGraphModule(torch.fx.GraphModule):
    def __init__(self, root: Union[torch.nn.Module, Dict[str, Any]], graph: torch.fx.Graph, class_name: str = "GraphModule"):
        super().__init__(root, graph, class_name)

    def delete_parameter(self, target: str) -> bool:
        atoms = target.split(".")
        path, target_param = atoms[:-1], atoms[-1]
        mod: torch.nn.Module = self

        # Get the parent module
        for item in path:
            if not hasattr(mod, item):
                return False

            mod = getattr(mod, item)

            if not isinstance(mod, torch.nn.Module):
                return False

        if not hasattr(mod, target_param):
            return False

        if not isinstance(getattr(mod, target_param), torch.nn.Parameter):
            return False

        delattr(mod, target_param)
        return True

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(3,2)
        self.linear2 = torch.nn.Linear(2,1)
    def forward(self, x):
        return self.linear2(self.linear1(x))


class DeeperModel(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nested_model = MyModel()
    def forward(self, x):
        return self.nested_model(x)


class StochasticTransformer(torch.fx.Transformer):
    def __init__(self, module: torch.fx.GraphModule):
        super().__init__(module)

    def my_transform(self, transform_list = None, transformation_fn = None):
        param_materialized = list(parameters_to_transform(self.module, transform_list))
        for name, param in param_materialized:
            # Transformation logic here
            new_param = GaussianParameter(shape = param.shape, mu=param)
            self.module.delete_parameter(name)
            self.module.add_submodule(name, new_param)





def get_nested_module(model, attr_path):
    attrs = attr_path.split('.')
    current = model
    for attr in attrs:
        current = getattr(current, attr)
    return current

def parameters_to_transform(model, transform_list = None):
    if transform_list is None:
        # Transform every parameters
        yield from model.named_parameters()
    else:
        # Transform only specified parameters
        for transform_name in transform_list:
            obj_to_transform = get_nested_module(model, transform_name)
            if isinstance(obj_to_transform, torch.nn.Module):
                yield from obj_to_transform.named_parameters(prefix=transform_name)
            elif isinstance(obj_to_transform, torch.nn.Parameter):
                yield transform_name, obj_to_transform


model = DeeperModel()
transform_list = ['nested_model.linear1',
                  'nested_model.linear2.bias']



#################
### Custom Tracer
#################

class CustomTracer(torch.fx.Tracer):
    def __init__(self):
        super().__init__()
        print("Initializing CustomTracer")

    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        print(f"Checking if {module_qualified_name} is a leaf module.")
        print(m.__class__)
        # Treat custom classes as leaves to avoid tracing their internals
        if isinstance(m, (Distribution)):
            return True

        # Default behavior for standard `torch.nn` modules
        return False
    


tracer = CustomTracer()
print(tracer.__dict__)
# param = torch.randn(784, 10)
# prova_trace = GaussianParameter(shape = param.shape, mu=param)

# class ToyModel(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.register_module('g_param',prova_trace)
    
#     def forward(self, n_samples, *args, **kwargs):
#         return self.g_param(n_samples)

# #print(isinstance(prova_trace, GaussianParameter))
# toy_prova = ToyModel()
# graph = tracer.trace(toy_prova)
# print(graph)
# print(toy_prova(25).shape)
# prova = StochasticTransformer(gm)
# prova.my_transform()

# print(list(gm.named_parameters()))




# # prova.my_transform()
# for name, param in gm.named_parameters():
#     atoms = name.split(".")
#     *path, target_param = atoms
#     path_name = ".".join(path)

#     if [path_name, target_param] in param_list:
#         print("Yes", name)
#     #     print("Path:", path)
#     #     print("TargetParam:", target_param)
#     #     print(param)
    



