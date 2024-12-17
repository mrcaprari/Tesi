import torch
from abc import abstractmethod


def torch_randn(*args, **kwargs):
    return torch.randn(*args, **kwargs)


class Gaussian(torch.nn.Module):
    def __init__(self, mu = None, std = None, register_fn = None):
        torch.nn.Module.__init__(self)  # Initialize nn.Module explicitly to avoid messing MRO
        if mu is None: 
            mu = torch.zeros(1)
        if std is None: 
            std = torch.ones(1)
        rho = torch.log(torch.exp(std) - 1)  # Inverse of softplus

        self.register_fn('mu', mu)
        self.register_fn('rho', rho)

    @property
    def std(self):
        return torch.nn.functional.softplus(self.rho)

    def forward(self, n_samples):
        epsilon = torch.randn(n_samples, *self.shape)
        return self.mu + epsilon * self.std
    
class Particle(torch.nn.modules.lazy.LazyModuleMixin, torch.nn.Module):
    def __init__(self, prior):
        super().__init__()
        self.register_module("prior", prior)
        self.particles = torch.nn.UninitializedParameter()

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
    


class Distribution(torch.nn.Module):
    def __init__(self, shape):
        torch.nn.Module.__init__(self)  # Initialize nn.Module explicitly to avoid MRO
        self.shape = shape
        # checks...

class Prior(Distribution):
    def __init__(self, shape):
        super().__init__(shape)
    
    def register_fn(self, name, attribute):
        return self.register_buffer(name, attribute)

class RandomParameter(Distribution):
    def __init__(self, shape, prior=None):
        super().__init__(shape)
        self.register_module('prior', prior)
    
    def register_fn(self, name, attribute):
        if attribute.numel() == 1:
            attribute = torch.full(self.shape, attribute.item())
        return self.register_parameter(name, torch.nn.Parameter(attribute))


##############################################################################
##############################################################################
##############################################################################
##############################################################################
class DistributionFactory:
    pass

class GaussianPrior(Gaussian, Prior):
    def __init__(self, shape, mu = None, std = None):
        Prior.__init__(self, shape)
        Gaussian.__init__(self, mu, std, register_fn = self.register_fn)

class GaussianParameter(Gaussian, RandomParameter):
    def __init__(self, shape, mu = None, std = None, prior = None):
        RandomParameter.__init__(self, shape, prior)
        Gaussian.__init__(self, mu, std, register_fn = self.register_fn)

    
# class GaussianDistribution(Gaussian, Distribution):
#     def __init__(self, shape, mu = None, std = None):
#         Distribution.__init__(self, shape)
#         Gaussian.__init__(self, mu, std)

#     def register_attribute(self, name, attribute):
#         return self.register_parameter(name, torch.nn.Parameter(attribute))

# prova = GaussianPrior(shape = ((10,5)))
# prova2 = GaussianParameter(shape = ((10,5)))
# particle = Particle(prova)

# print("GaussianPrior:")
# print("Parameters:")
# print(list(prova.named_parameters()))
# print("Buffers:")
# print(list(prova.named_buffers()))

# print("\n")
# print("GaussianParameter:")
# print("Parameters:")
# print(list(prova2.named_parameters()))
# print("Buffers:")
# print(list(prova2.named_buffers()))

# particle(1)
# print("\n")
# print("GaussianParticle:")
# print("Parameters:")
# print(list(particle.named_parameters()))
# print("Buffers:")
# print(list(particle.named_buffers()))

