import torch
from abc import abstractmethod
from torch.nn.modules.lazy import LazyModuleMixin

class DistributionRole(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)  

class Prior(DistributionRole):
    def __init__(self):
        super().__init__()
    
    def register_fn(self, name, attribute):
        return self.register_buffer(name, attribute)

class RandomParameter(DistributionRole):
    def __init__(self, prior):
        super().__init__()
        # Ensure buffers in prior are initialized
        self.register_module("prior", prior)
    
    def register_fn(self, name, attribute):
        if attribute.numel() == 1:
            attribute = torch.full(self.shape, attribute.item())
        return self.register_parameter(name, torch.nn.Parameter(attribute))


class Particle(torch.nn.modules.lazy.LazyModuleMixin, DistributionRole):
    def __init__(self, prior):
        super().__init__()
        self.register_module("prior", prior)
        self.register_parameter("particles", torch.nn.UninitializedParameter())

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
        
    @property
    def flattened_particles(self):
            return torch.flatten(self.particles, start_dim=1)
    
    def forward(self, *args, **kwargs):
        return self.particles