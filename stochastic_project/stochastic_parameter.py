import torch

class StochasticParameter(torch.nn.Module):
    def __init__(self, mu: torch.Tensor, rho: torch.Tensor):
        super().__init__()
        if mu.shape != rho.shape:
            raise ValueError(f"Shape mismatch: mu {mu.shape} and rho {rho.shape} must match.")
        self.mu = torch.nn.Parameter(mu, requires_grad=True)  # Store mean as a parameter
        self.rho = torch.nn.Parameter(rho, requires_grad=True)  # Store rho as a parameter

    @classmethod
    def from_mu_std(cls, mu: torch.nn.Parameter, std: torch.nn.Parameter):
        if mu.shape != std.shape:
            raise ValueError(f"Shape mismatch: mu {mu.shape} and std {std.shape} must match.")
        rho = torch.log(torch.exp(std) - 1)  # Inverse of softplus
        return cls(mu, rho)

    @classmethod
    def from_mu(cls, mu: torch.nn.Parameter):
        std = torch.ones_like(mu)  # Default std
        return cls.from_mu_std(mu, std)

    @classmethod
    def from_shape(cls, shape: torch.Size, device=None):
        mu = torch.zeros(shape, device=device, requires_grad=True)
        return cls.from_mu(mu)

    @property
    def std(self):
        return torch.nn.functional.softplus(self.rho)

    def forward(self, n_samples, *args, **kwargs):
        epsilon = torch.randn((n_samples, *self.shape), device=self.mu.device)
        return self.mu + self.std * epsilon

    @property
    def shape(self):
        return self.mu.shape