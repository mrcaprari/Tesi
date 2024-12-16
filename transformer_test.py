import torch.fx
import torch
from torch.nn.modules.lazy import LazyModuleMixin
#from stochastic_project.stochastic_parameter.gaussian_parameter import GaussianParameter


################
#### IMPORT ####
################

class GaussianDistribution(torch.nn.Module):
    def __init__(self, mu=None, std=None, shape=None, register_fn=None):
        super().__init__()
        
        shape = self._infer_shape(mu, std, shape)
        self.shape = torch.Size(shape)

        # Initialize mu parameter with fallback default
        register_fn(self, "mu", self._initialize_param(mu, shape, default=torch.zeros(1)))

        # Initialize rho parameter (inferred from std) with fallback default
        std = self._initialize_param(std, shape, default=torch.ones(1))
        rho = torch.log(torch.exp(std) - 1)  # Inverse of softplus
        register_fn(self,"rho", rho)
        
    def _infer_shape(self, mu, std, shape):
        """
        Infers the shape from mu or std if not explicitly provided.
        """
        if shape is not None:
            return shape  # Use provided shape
        if mu is not None:
            try: return mu.shape
            except: ValueError("mu has no 'shape' attribute")
        if std is not None:
            try: return std.shape
            except: ValueError("std has no 'shape' attribute")
        raise ValueError("At least one of 'mu', 'std', or 'shape' must be specified.")
        
    def _initialize_param(self, param, shape, default):
        if param is None:
            return default
        if param.shape == shape:
            return param
        if param.numel() == 1:
            return torch.full(self.shape, param.item())
        raise ValueError(f"{param.numel()} shape different from distribution shape {self.shape}" )

    @property
    def std(self):
        return torch.nn.functional.softplus(self.rho)

    def forward(self, n_samples):
        epsilon = torch.randn(n_samples, *self.shape)
        return self.mu + self.std * epsilon

    def __repr__(self):
        return f"GaussianDistribution(mu={self.mu},\n std={self.std})"


class GaussianPrior(GaussianDistribution):
    def __init__(self, mu=None, std=None, shape=None):
        super().__init__(mu, std, shape, register_fn=self._register_buffer)

    def _register_buffer(self, module, name, param):
        module.register_buffer(name, param)


class GaussianParameter(GaussianDistribution):
    def __init__(self, mu=None, std=None, shape=None, prior=None):
        if prior is not None:
            # If a prior is provided, extract its parameters
            if not isinstance(prior, GaussianPrior):
                raise TypeError("'prior' must be an instance of GaussianPrior.")
            mu = prior.mu if mu is None else mu
            std = prior.std if std is None else std
            shape = prior.shape if shape is None else shape
    
        super().__init__(mu, std, shape, register_fn=self._register_parameter)
        self.prior = prior or GaussianPrior(shape=self.shape)

    def _register_parameter(self, module, name, param):
        if param.shape != self.shape:
            print(f"Shape of {name} changed from {param.shape} to {self.shape}")
            param = torch.full(self.shape, param.item())
        module.register_parameter(name, torch.nn.Parameter(param))

    def __repr__(self):
        return (f"GaussianParameter(mu_shape={self.mu.shape}, std_shape={self.std.shape}, "
                f"prior={repr(self.prior)})")


class GaussianParticles(torch.nn.modules.lazy.LazyModuleMixin, torch.nn.Module):
    def __init__(self, prior = None):
        super().__init__()  # Initialize LazyModuleMixin and Module
        if prior is None or not isinstance(prior, GaussianPrior):
            raise TypeError("'prior' must be an instance of GaussianPrior.")
        self.prior = prior
        self.particles = torch.nn.UninitializedParameter()  # Lazy initialization of particles
    
    def initialize_parameters(self, n_particles):
        """Materialize and initialize particles based on the prior."""
        if self.has_uninitialized_params():
            # Materialize the parameter with the correct shape
            self.particles.materialize((n_particles, *self.prior.shape))

        # Initialize particles using the prior's forward method
        with torch.no_grad():
            self.particles = torch.nn.Parameter(
                self.prior.forward(n_particles)
            )

    def forward(self, *args, **kwargs):
        return self.particles
    
    @property
    def flattened_particles(self):
            return torch.flatten(self.particles, start_dim=1)




########

class CustomTracer(torch.fx.Tracer):
    def __init__(self, transform_list = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transform_list = transform_list

    def is_leaf_module(self, module, module_qualified_name):
        # Treat all modules as non-leaf
        return False

    def create_node(self, kind, target, args, kwargs, name = None, type_expr = None):
        node = super().create_node(kind, target, args, kwargs, name, type_expr)
        if node.op == "get_attr": 
            if not self.transform_list:
                node.meta = {**node.meta, 'transform': True}
            elif any(node.target.startswith(transform) for transform in self.transform_list):
                node.meta = {**node.meta, 'transform': True}
            else:
                node.meta = {**node.meta, 'transform': False}
        if node.op =="call_function" and any(getattr(arg,'meta',{}).get('transform') for arg in node.args):
            node.meta = {**node.meta, 'transform': True}
        if node.op == "placeholder":
            self.last_placeholder = node

        return node
    
    def trace(self, root):
            graph = super().trace(root)
            return graph, self.last_placeholder



class MyGraphModuleFactory():
    def __init__(self, module, preliminary_graph, last_placeholder):
        self.module = module
        self.preliminary_graph = preliminary_graph
        self.last_placeholder = last_placeholder
    
    @property
    def new_graph(self):
        with self.preliminary_graph.inserting_after(self.last_placeholder):
            n_samples_node = self.preliminary_graph.placeholder(name = "n_samples")
    
        for node in self.preliminary_graph.nodes:
            if getattr(node,'meta',{}.get('transform', False)):
                if node.op == "get_attr":
                    self._transform_parameter(self.module, node.target)
                    node.op = "call_module"
                    node.args = (n_samples_node,)
                elif node.op == "call_function":
                    in_dims=tuple(0 if getattr(arg,'meta',{}).get('transform')
                                  else None
                                  for arg in node.args)
                    
                    node.meta = {**node.meta, 'in_dims': in_dims}
                
        return graph
    
    @property
    def new_module(self):
        return torch.fx.GraphModule(self.module, self.new_graph)

    def _transform_parameter(self, module, name):
            print(f"Transforming {name} parameter")
            attrs = name.split('.')
            *path, param_name = attrs
            submodule = module
            for attr in path:
                submodule = getattr(submodule, attr)
            param = getattr(submodule, param_name)            
            delattr(submodule, param_name)
            new_param = GaussianParameter(shape = param.shape, mu=param)
            submodule.register_module(param_name, new_param)
            return True

def vmap_wrapper(target, in_dims):
    def wrapped(*args, **kwargs):
        return torch.vmap(target, in_dims=in_dims)(*args, **kwargs)
    return wrapped

class MyTransformer(torch.fx.Transformer):
    def run_node(self, n):
        if n.op =="call_function":
            current_in_dims = getattr(n,'meta',{}).get('in_dims', None)
            self.in_dims = current_in_dims
        return super().run_node(n)
    
    def call_function(self, target, args, kwargs):
        if self.in_dims is not None:
            print("in_dims:",self.in_dims)
            print("target:", target)
            print("args:", args)
            print("kwargs:", kwargs)
            vmap_fn = vmap_wrapper(target, self.in_dims)

            # Pass the vmap-wrapped function as the target
            return super().call_function(vmap_fn, args, kwargs)
        return super().call_function(target, args, kwargs)
    
########################
###### PROVIAMOLO #######
########################

# Define a simple module
class SimpleModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    
    def forward(self, x):
        x = self.linear(x)
        x = torch.relu(x)  # This should invoke a call_function node
        x = torch.sum(x, dim=1)  # Another call_function node
        return x

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10,5)
        self.second = torch.nn.Linear(5,1)
    def forward(self,x):
        return self.second(self.linear(x))

# Instantiate and trace the module
simple_module = SimpleModule()
x = torch.randn(25, 10)
prova = MyModel()

my_tracer = CustomTracer()
graph, last_place = my_tracer.trace(prova)

graph_fact = MyGraphModuleFactory(prova, graph, last_place)
new_module = torch.fx.GraphModule(prova, graph_fact.new_graph)

trans_graph = MyTransformer(new_module).transform()

output = trans_graph(x, n_samples = 10)
print(output.shape)


# print(graph)
# print("\n")
# for node in graph.nodes:
#     print(node.name, node.meta)
