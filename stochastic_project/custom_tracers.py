import torch.fx

# class NoLeafOnly(torch.fx.Tracer):
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
#             else:
#                 node.meta = {**node.meta, 'transform': False}
        
#         return node


# class DeterministicTransformer(torch.fx.Transformer):
#     def run_node(self, n):
#         # Add your custom logic here if needed
#         if n.op =="get_attr" and getattr(n,'meta',{}).get('transform', False):
#             #print(n.meta)
#             self._transform_parameter(self.module, n.target)
#             n.target = f'{n.target}.mu'
#             n.name = f'{n.name}'
#             print(f"Running node: {n.format_node()}")  # Example: Log node execution

#         return super().run_node(n)

#     def _transform_parameter(self, module, name):
#             print(f"Transforming {name} parameter")
#             attrs = name.split('.')
#             *path, param_name = attrs
#             submodule = module
#             for attr in path:
#                 submodule = getattr(submodule, attr)
#             param = getattr(submodule, param_name)            
#             delattr(submodule, param_name)
#             new_param = GaussianParameter(shape = param.shape, mu=param)
#             submodule.register_module(param_name, new_param)
#             return True
    
#     def transform(self):
#         return super().transform()