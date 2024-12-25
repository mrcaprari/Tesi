import torch.fx
import torch

class PreparatoryTracer(torch.fx.Tracer):
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
        if node.op =="call_function" and any(arg.meta.get('transform', False) for arg in node.args):
            node.meta = {**node.meta, 'transform': True}

        return node