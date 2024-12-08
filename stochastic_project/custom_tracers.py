import torch.fx

class NoLeafTracer(torch.fx.Tracer):
    def is_leaf_module(self, module, module_qualified_name):
        # Treat all modules as non-leaf
        return False
