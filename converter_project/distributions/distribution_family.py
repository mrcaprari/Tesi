import math

import torch


class Gaussian:
    def __init__(self, mu=None, std=None, shape=None, register_fn=None):
        if mu is not None and std is not None:
            self._initialize(mu=mu, std=std)
        elif mu is not None:
            self._initialize(mu=mu)
        elif shape is not None:
            self._initialize(shape=shape)
        else:
            raise ValueError(
                "You must provide `mu`, `std`, or `shape` for initialization."
            )

    def _initialize(self, mu=None, std=None, shape=None, register_fn=None):
        self.shape = shape or mu.shape

        if mu is None:
            mu = torch.zeros(1)
        if std is None:
            std = torch.ones(1)
        rho = torch.log(torch.exp(std) - 1)  # Inverse of softplus

        self.register_fn("mu", mu)
        self.register_fn("rho", rho)

    @classmethod
    def from_param(cls, param):
        if isinstance(param, (torch.nn.Parameter, torch.Tensor)):
            return cls(mu=param)

        return cls(mu=param.mu, std=param.std)

    @property
    def std(self):
        return torch.nn.functional.softplus(self.rho)

    def log_prob(self, value):
        var = self.std**2
        log_scale = self.std.log()

        return (
            -((value - self.mu) ** 2) / (2 * var)
            - log_scale
            - 0.5 * math.log(2 * math.pi)
        )

    def forward(self, n_samples):
        epsilon = torch.randn(n_samples, *self.shape)
        return self.mu + epsilon * self.std
