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

    def general_product(self, A, B):
        assert A.shape[0] == A.shape[1], "A must be a square matrix"
        assert A.shape[1] == B.shape[0], "The first dimension of B must match A's dimensions"

        # Create the einsum equation dynamically
        ndim_B = B.ndim
        suffix = ''.join([chr(ord('k') + i) for i in range(ndim_B - 1)])
        equation = f"ij,j{suffix}->i{suffix}"
        # Perform the product
        return torch.einsum(equation, A, B)

    def perturb_gradients(self, kernel_matrix):
        self.particles.grad = self.general_product(kernel_matrix, self.particles.grad)

    def forward(self, *args, **kwargs):
        return self.particles