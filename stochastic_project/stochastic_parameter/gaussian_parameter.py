import torch
from torch.nn.modules.lazy import LazyModuleMixin

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





# class Distribution():
#     def __init__(self, shape, **kwargs):
#         self.shape = self._infer_shape(shape, kwargs)

#     def _infer_shape(self, shape, *args):
#         if shape is not None:
#             return shape  # Use provided shape
#         for arg in args:
#             if arg is not None:
#                 return arg.shape
#         raise ValueError("Specify a 'shape' or provide an argument with 'shape' attribute")

# class Prior(Distribution):
#     def __init__(self, shape, **kwargs):
#         super().__init__(shape, **kwargs)
#         for name, param in kwargs:
#             self.register_buffer(name, param)


# class Gaussian(torch.nn.Module):
#     def __init__(self, mu, std):        
#         if mu is None:
#             mu = torch.zeros(1)
#         if std is None:
#             std = torch.ones(1)

#     def forward(self, n_samples):
#         epsilon = torch.randn(n_samples, self.shape)
#         return self.mu + epsilon + self.mu


# class RandomParameter(Distribution):
#     def __init__(self, *args, prior: Prior):
#         super().__init__()
#         self.register_module("prior", prior)

#     def _register_attr(self):
#         return self.register_parameter()
