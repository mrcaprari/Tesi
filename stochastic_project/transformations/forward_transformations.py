import torch
import torch.fx

def vmap_forward_transformer(module):
    def vmap_wrapper(target, in_dims):
        def wrapped(*args, **kwargs):
            return torch.vmap(target, in_dims=in_dims)(*args, **kwargs)
        return wrapped

    class ForwardTransformer(torch.fx.Transformer):
        def run_node(self, n):
            if n.op =="call_function":
                current_in_dims = n.meta.get('in_dims', None)
                self.in_dims = current_in_dims
            return super().run_node(n)
        
        def call_function(self, target, args, kwargs):
            if self.in_dims is not None:
                vmap_fn = vmap_wrapper(target, self.in_dims)
                return super().call_function(vmap_fn, args, kwargs)
            return super().call_function(target, args, kwargs)
    
    return ForwardTransformer(module).transform()


# def vmap_wrapper(target, in_dims):
#     def wrapped(*args, **kwargs):
#         return torch.vmap(target, in_dims=in_dims)(*args, **kwargs)
#     return wrapped

# class ForwardTransformer(torch.fx.Transformer):
#     def run_node(self, n):
#         if n.op =="call_function":
#             current_in_dims = n.meta.get('in_dims', None)
#             self.in_dims = current_in_dims
#         return super().run_node(n)
    
#     def call_function(self, target, args, kwargs):
#         if self.in_dims is not None:
#             vmap_fn = vmap_wrapper(target, self.in_dims)
#             return super().call_function(vmap_fn, args, kwargs)
#         return super().call_function(target, args, kwargs)