import torch
import torch.fx

# Register the wrapped function with FX
def vmap_builder(target_fn, current_in_dims):
    return torch.vmap(func=target_fn, 
                      in_dims=current_in_dims)

torch.fx.wrap("vmap_builder")


class NewVmapApplicator(torch.fx.Transformer):
    def __init__(self, module: torch.nn.Module):
        print("Inside NewVmap")
        super().__init__(module)
    
    def run_node(self, n):

        if n.op == "call_function":

            self.current_in_dims = tuple(
                0 if getattr(arg, 'meta', {}).get('additional_map_dim', False) else None
                for arg in n.args
            )

        else:
            self.current_in_dims = None
        return super().run_node(n)

    def call_function(self, target, args, kwargs):
        if any(dim is not None for dim in self.current_in_dims):
            current_in_dims = self.current_in_dims

            def vmap_fn(*vmap_args, **vmap_kwargs):
                return vmap_builder(target_fn=target, 
                                    current_in_dims=current_in_dims)(*vmap_args, **vmap_kwargs)

            # Create a proxy node for vmap_fn
            return self.tracer.create_proxy(
                'call_function', vmap_fn, args, kwargs
            )

        return super().call_function(target, args, kwargs)
