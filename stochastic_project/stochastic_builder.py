import torch.fx
from torch.fx import wrap
from stochastic_parameter import *

def all_users(node: torch.fx.Node) -> set:
    users = set()
    def collect_users(n):
        for user in n.users:
            if user not in users:
                users.add(user)
                collect_users(user)
    collect_users(node)
    return users

@wrap
def apply_vmap(target_fn, in_dims, *args, **kwargs):
    print(f"Applying vmap to {target_fn} with args: {args}, kwargs: {kwargs}")
    return torch.vmap(func=target_fn, in_dims=in_dims)(*args, **kwargs)


#torch.fx.wrap("vmap_builder")

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10,5)
        self.second = torch.nn.Linear(5,1)
    def forward(self,x):
        return self.second(self.linear(x))


# class NoLeafTracer(torch.fx.Tracer):
#     def __init__(self, transform_list, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.transform_list = transform_list

#     def is_leaf_module(self, module, module_qualified_name):
#         # Treat all modules as non-leaf
#         return False
    
#     def create_node(self, kind, target, args, kwargs, name = None, type_expr = None):
#         node = super().create_node(kind, target, args, kwargs, name, type_expr)

#         if node.op == "get_attr": 
#             if not self.transform_list:
#                 node.meta = {**node.meta, 'transform': True}
#             elif any(node.target.startswith(transform) for transform in self.transform_list):
#                 node.meta = {**node.meta, 'transform': True}
        
#         elif node.op =="call_function":
#             in_dims = tuple(0 if getattr(arg,'meta',{}).get('transform', False) else None
#                             for arg in node.args)
            
#             if any(in_dim is not None for in_dim in in_dims):
#                 new_target = vmap_builder(target_fn = node.target,
#                                         in_dims = in_dims)
#                 node.target = new_target
#                 node.meta = {**node.meta, 'transform': True}
#                 print(node.name, in_dims, node.target, new_target)
    
#         return node
    


# class StochasticBuilder():
#     def __init__(self, transform_list = [], transform_logic=None):
#         self.tracer = NoLeafTracer(transform_list)
#         self.transform_list = transform_list
#         self.transform_logic = transform_logic
    
#     def prepare_graph(self, module):
#         self.original_graph = self.tracer.trace(module)
    
#     def transform(self, module):
#         self.prepare_graph(module)
#         gm = torch.fx.GraphModule(module, self.original_graph)

#         return self._transform_parameters(gm)


#     def _transform_parameters(self, gm):
#         param_materialized = list(self._parameters_to_transform(gm))
#         for name, param in param_materialized:
#             # Transformation logic here
#             new_param = GaussianParameter(shape = param.shape, mu=param)
#             self._delete_parameter(gm, name)
#             gm.add_submodule(name, new_param)
#         return gm

#     def _delete_parameter(self, module, target: str) -> bool:
#         atoms = target.split(".")
#         path, target_param = atoms[:-1], atoms[-1]
#         mod = module
#         for item in path:
#             if not hasattr(mod, item):
#                 return False
#             mod = getattr(mod, item)
#             if not isinstance(mod, torch.nn.Module):
#                 return False
#         if not hasattr(mod, target_param):
#             return False
#         if not isinstance(getattr(mod, target_param), torch.nn.Parameter):
#             return False
#         delattr(mod, target_param)
#         return True

#     def _parameters_to_transform(self, model):
#         if not self.transform_list:
#             # Transform every parameters
#             yield from model.named_parameters()
#         else:
#             # Transform only specified parameters
#             for transform_name in self.transform_list:
#                 obj_to_transform = self._get_nested_module(model, transform_name)
#                 if isinstance(obj_to_transform, torch.nn.Module):
#                     yield from obj_to_transform.named_parameters(prefix=transform_name)
#                 elif isinstance(obj_to_transform, torch.nn.Parameter):
#                     yield transform_name, obj_to_transform

#     def _get_nested_module(self, model, attr_path):
#         attrs = attr_path.split('.')
#         current = model
#         for attr in attrs:
#             current = getattr(current, attr)
#         return current

# prova = MyModel()




# tracer = NoLeafTracer(transform_list)
# graph = tracer.trace(prova)
# for node in graph.nodes:
#     print(node, node.name, node.op, node.target, )







# transform = StochasticBuilder(transform_list)
# prova = transform.transform(prova)


# for node in prova.graph.nodes:
#     if node.op =="get_attr":
#         if getattr(node,'meta', {}).get('transform', False):
#             pass

#     elif node.op =="call_function":
#         in_dims = tuple(0 if getattr(arg,'meta',{}).get('transform', False) else None
#                         for arg in node.args)
        
#         if any(in_dim is not None for in_dim in in_dims):
#             node.target = vmap_builder(target_fn = node.target,
#                                        in_dims = in_dims)
#             node.meta = {**node.meta, 'transform': True}
#             print(node.name, in_dims, node.target)





#########################################################################
#########################################################################
#########################################################################
#########################################################################

class NoLeafOnly(torch.fx.Tracer):
    def __init__(self, transform_list, *args, **kwargs):
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
        
        if node.op == "placeholder":
            self.last_placeholder = node

        if node.op =="call_function":
            current_in_dims = tuple(
                0 if getattr(arg,'meta',{}).get('transform', False) else None
                for arg in node.args
            )
            if any(in_dim is not None for in_dim in current_in_dims):
                print(current_in_dims)
                print(*node.args)
                node.target = self._apply_vmap(node.target, current_in_dims, *args, **kwargs)

#                vmap_fn = torch.vmap(node.target, in_dims = current_in_dims)
                
        return node
        
    def _apply_vmap(self, target_fn, in_dims, *args, **kwargs):
        # Applies vmap to the given function
        print(f"Applying vmap to {target_fn} with args: {args}, kwargs: {kwargs}")
        return torch.vmap(func=target_fn, in_dims=in_dims)(*args, **kwargs)

    def trace(self, root):
        """
        Extend the trace method to add the n_samples node.
        """
        # Start tracing by calling the parent class's trace method
        graph = super().trace(root)

        # Add the n_samples node to the graph
        with graph.inserting_after(self.last_placeholder):
            n_samples_node = graph.create_node(
                op="placeholder",  # This makes it a placeholder input node
                target="n_samples",  # The name of the node
                args=(),             # No arguments for the placeholder
                kwargs={'default':1},
                name="n_samples"     # Explicit name for clarity
            )
        
        for node in self.graph.nodes:
            if getattr(node,'meta',{}).get('transform', False):
                new_args = (*node.args, n_samples_node)
                node.args = new_args

    
        return graph
       
        
        # elif node.op =="call_function":
        #     in_dims = tuple(0 if getattr(arg,'meta',{}).get('transform', False) else None
        #                     for arg in node.args)

        #     node.meta = {**node.meta, 'in_dims': in_dims}
        #     if any(in_dim is not None for in_dim in in_dims):
        #         node.meta = {**node.meta, 'add_dim': True}

            # if any(in_dim is not None for in_dim in in_dims):
            #     new_target = vmap_builder(target_fn = node.target,
            #                             in_dims = in_dims)
            #     node.target = new_target
            #     node.meta = {**node.meta, 'transform': True}
            #     print(node.name, in_dims, node.target, new_target)




class StochasticTransformer(torch.fx.Transformer):
    def run_node(self, n):
        # Add your custom logic here if needed
        if n.op =="get_attr" and getattr(n,'meta',{}).get('transform', False):
            #print(n.meta)
            self._transform_parameter(self.module, n.target)
            n.op = "call_module"           
            print(f"Running node: {n.format_node()}")  # Example: Log node execution

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

    


x = torch.randn(25, 10)
prova = MyModel()
output1 = prova(x)


transform_list = ['linear']
custom_tracer = NoLeafOnly(transform_list=transform_list)

prova_graph = custom_tracer.trace(prova)
print(prova_graph)
prova_gm = torch.fx.GraphModule(prova, prova_graph)
s_prova = StochasticTransformer(prova_gm).transform()

print(s_prova(x, n_samples = 2))
#print(prova_graph.find_nodes(op = "placeholder")[-1])
#print(prova_gm.graph)

#print(s_prova(x))


#print(torch.allclose(output1,det_prova(x, n_samples = 1)))
# print(list(prova_gm.named_parameters()))

#output2 = prova_gm(x)
#print(torch.allclose(output1, output2))
###13_01_16/12
# class RandomGraphFactory():
#     def __init__(self, mode, tracer = NoLeafOnly()):
#         self.mode = mode
#         self.tracer = tracer
    
#     def register_original_graph(self, module):
#         self.original_graph = self.tracer.trace(module)

#     def random_module(self, module, transform_list):
#         self.register_original_graph(module)
#         self.transform_list = transform_list
#         old_gm = torch.fx.GraphModule(module, self.original_graph)
#         random_gm = self._transform_parameters(old_gm)
#         random_graph = self.create_new_graph(module, transform_list)
#         return random_graph, random_gm

#         #return torch.fx.GraphModule(random_gm, random_graph)
        
#     def create_new_graph(self, module, transform_list=None):
#         new_graph = torch.fx.Graph()
#         new_node_map = {}

#         last_placeholder = None

#         for node in self.original_graph.nodes:
#             if node.op == "placeholder":
#                 new_placeholder = new_graph.create_node(
#                     op="placeholder",
#                     target=node.target,
#                     args=(),
#                     kwargs={},
#                     name=node.name
#                 )
#                 new_node_map[node] = new_placeholder
#                 # Remember place of last_placeholder 
#                 last_placeholder = new_placeholder

#             # We add n_samples placeholder after each other placeholder
#         with new_graph.inserting_after(last_placeholder):
#             n_samples_node = new_graph.placeholder(name="n_samples", default_value=1)

#         for node in self.original_graph.nodes:
#             if node.op == "placeholder":
#                 continue

#             in_dims = tuple(
#                 0 if getattr(arg,'meta',{}).get('add_dim', False) else None 
#                             for arg in node.args)            
        
#             if node.op == "get_attr" and any(node.target.startswith(transform) for transform in transform_list): 
#                 new_node = new_graph.create_node(
#                     op = "call_module",
#                     target = node.target,
#                     args= (n_samples_node,),
#                     name=f'random_{node.name}'
#                 )
#                 node.meta = {**node.meta, 'add_dim': True}
#                 new_node.meta = {**node.meta, 'add_dim': True}
#                 new_node_map[node] = new_node

#             elif node.op =="call_function" and any(in_dim is not None for in_dim in in_dims): 
#                 current_in_dims = in_dims        
#                 target_fn = node.target           
#                 new_args = tuple(new_node_map[arg] for arg in node.args)
#                 # print("Node:", node.name)
#                 # print("Args:", new_args)
#                 # print("In dims:", in_dims)

#                 # def vmap_fn(*args, **kwargs):
#                 #     print("Inside_vmap:", current_in_dims, target_fn)
#                 #     return torch.vmap(func=target_fn, 
#                 #                       in_dims=current_in_dims)(*args, **kwargs)

#                 new_node = new_graph.create_node(
#                     op = "call_function",
#                     target = apply_vmap,
#                     args= (target_fn, current_in_dims, *new_args),
#                     kwargs= node.kwargs,
#                     name = f"{node.name}_vmap"
#                 )

#                 node.meta = {**node.meta, 'add_dim': True}
#                 new_node.meta = {**node.meta, 'add_dim': True}
#                 new_node_map[node] = new_node

#             else:
#                 new_node = new_graph.node_copy(node, lambda n :new_node_map[n])
#                 new_node_map[node] = new_node
    
#         return new_graph
    
#     def _transform_parameters(self, gm):
#         param_materialized = list(self._parameters_to_transform(gm))
#         for name, param in param_materialized:
#             # Transformation logic here
#             new_param = GaussianParameter(shape = param.shape, mu=param)
#             self._delete_parameter(gm, name)
#             gm.add_submodule(name, new_param)
#         return gm

#     def _delete_parameter(self, module, target: str) -> bool:
#         atoms = target.split(".")
#         path, target_param = atoms[:-1], atoms[-1]
#         mod = module
#         for item in path:
#             if not hasattr(mod, item):
#                 return False
#             mod = getattr(mod, item)
#             if not isinstance(mod, torch.nn.Module):
#                 return False
#         if not hasattr(mod, target_param):
#             return False
#         if not isinstance(getattr(mod, target_param), torch.nn.Parameter):
#             return False
#         delattr(mod, target_param)
#         return True

#     def _parameters_to_transform(self, model):
#         if not self.transform_list:
#             # Transform every parameters
#             yield from model.named_parameters()
#         else:
#             # Transform only specified parameters
#             for transform_name in self.transform_list:
#                 obj_to_transform = self._get_nested_module(model, transform_name)
#                 if isinstance(obj_to_transform, torch.nn.Module):
#                     yield from obj_to_transform.named_parameters(prefix=transform_name)
#                 elif isinstance(obj_to_transform, torch.nn.Parameter):
#                     yield transform_name, obj_to_transform

#     def _get_nested_module(self, model, attr_path):
#         attrs = attr_path.split('.')
#         current = model
#         for attr in attrs:
#             current = getattr(current, attr)
#         return current



# prova = MyModel()
# transform_list = ['linear.weight']
# factory = RandomGraphFactory(mode="deterministic")



# random_graph, random_gm = factory.random_module(prova, transform_list)
# cosa = torch.fx.Interpreter(random_gm, graph=random_graph)
# # print(random_graph)
# # print(random_gm.code)
# print(random_graph._codegen)
#print(new_gm.graph)
#print(random.code)  

#x = torch.randn(25, 5)

#print(random(x, 1).shape)




#### DETERMINISTIC GRAPH
            # if node.op != "placeholder":
            #     if node.op == "get_attr" and any(node.target.startswith(transform) for transform in transform_list): 
            #         new_node = new_graph.create_node(
            #             op = "get_attr",
            #             target = f'{node.target}.mu'
            #         )
            #         node.meta = {**node.meta, 'add_dim': False}
            #         new_node_map[node] = new_node

####RANDOM PARAMETER
            # if node.op != "placeholder":
            #     if node.op == "get_attr" and any(node.target.startswith(transform) for transform in transform_list): 
            #         new_node = new_graph.create_node(
            #             op = "call_module",
            #             target = node.target,
            #             args=n_samples_node
            #         )
            #         node.meta = {**node.meta, 'add_dim': False}
            #         new_node_map[node] = new_node

        # for node in self.original_graph.nodes:
        #     if node.op != "placeholder":
        #         if node.op == "get_attr":
        #             # Switch specified parameters with stochastic parameters                 
        #             if not self.transform_list:
        #                 new_node = new_graph.create_node(
        #                     op = "",
        #                     target = node.target,
        #                     args = (),
        #                     kwargs = {},
        #                     name = node.name
        #                 )
        #                 new_node.meta = {**node.meta, 'transform': mode}

        #             elif any(node.target.startswith(transform) for transform in self.transform_list):
        #                 node.meta = {**node.meta, 'transform': True}

        #         if node in self.stochastic_nodes:
        #             # Create the first node pointing to the submodule
        #             submodule_node = new_graph.create_node(
        #                 op="get_attr",
        #                 target=node.target,
        #             )

        #             # Create the second node accessing the "mu" parameter of the submodule
        #             realization_node = new_graph.create_node(
        #                 op="call_module",
        #                 target=f"{node.target}",
        #                 args=(n_samples_node,)
        #             )

        #             realization_node.meta = {**node.meta, 'additional_map_dim': True}

        #             for user in utils.all_users(node):
        #                 user.meta = {**user.meta, 'additional_map_dim': True}

        #             BBVI_node_map[user] = user
        #             BBVI_node_map[node] = realization_node
        #         else:
        #             new_node = BBVI_graph.node_copy(node,
        #                                                 lambda n: BBVI_node_map[n])
        #             new_node.meta = {**node.meta}
        #             BBVI_node_map[node] = new_node

        # return BBVI_graph




# for node in prova.graph.nodes:
#     if node.op == "call_function":
#         print(f"Node {node.name}: Target = {node.target}")

# # # Ensure the graph structure is consistent after changes
# prova.graph.lint()
# print(prova.graph)

# for name, param in prova.named_parameters():
#     print(name, param)

# for node in transform.original_graph.nodes:
#     if getattr(node,'meta', {}).get('transform', False):
#         print(node.target)


    # def transform(self, module):
    #     original_graph = self.tracer.trace(module)
    #     for transform in self.transform_list:
    #         for node in original_graph.nodes:
    #             if node.target is transform:
    #                 print(node.target)
    #                 break
            
    #         transform_node = original_graph.find_nodes(op="get_attr", target=transform)
            
    #         print(transform_node)

        # original_graph.find_nodes()
        #     if node.op == "get_attr":
        #         # Exact match for specific parameters
        #         if node.target in self.transform_list:
        #             print(f"Matched specific parameter: {node.target}")
                

        #         # Match submodules by prefix
        #         for submodule in self.transform_list:
        #             if isinstance(node.target, str) and node.target.startswith(submodule):
        #                 print(f"Matched submodule parameter: {node.target}")



