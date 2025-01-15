from typing import Optional, Union

import torch


class Gaussian:
    """
    A class representing a Gaussian distribution as a torch.nn.Module
    to define a probabilistic distribution with trainable parameters
    suitable for automatic differentiation and the reparameterization trick.

    Attributes:
        shape (torch.Size): The shape of the distribution.
        mu (torch.Tensor): The mean of the Gaussian distribution.
        rho (torch.Tensor): The parameter related to standard deviation via softplus.
    """

    def __init__(
        self,
        mu: Optional[torch.Tensor] = None,
        std: Optional[torch.Tensor] = None,
        shape: Optional[torch.Size] = None,
        register_fn: Optional[callable] = None,
    ) -> None:
        """
        Initialize the Gaussian distribution.

        Args:
            mu (Optional[torch.Tensor]): Mean of the distribution.
            std (Optional[torch.Tensor]): Standard deviation of the distribution.
            shape (Optional[torch.Size]): Shape of the distribution if `mu` and `std` are not provided.
            register_fn (Optional[callable]): Function to register `mu` and `rho` as parameters or buffers.

        Raises:
            ValueError: If none of `mu`, `std`, or `shape` are provided.
        """
        if mu is not None and std is not None:
            self._initialize(mu=mu, std=std, register_fn=register_fn)
        elif mu is not None:
            self._initialize(mu=mu, register_fn=register_fn)
        elif shape is not None:
            self._initialize(shape=shape, register_fn=register_fn)
        else:
            raise ValueError(
                "You must provide `mu`, `std`, or `shape` for initialization."
            )

    def _initialize(
        self,
        mu: Optional[torch.Tensor] = None,
        std: Optional[torch.Tensor] = None,
        shape: Optional[torch.Size] = None,
        register_fn: Optional[callable] = None,
    ) -> None:
        """
        Helper method to initialize the Gaussian distribution.

        Args:
            mu (Optional[torch.Tensor]): Mean of the distribution.
            std (Optional[torch.Tensor]): Standard deviation of the distribution.
            shape (Optional[torch.Size]): Shape of the distribution.
            register_fn (Optional[callable]): Function to register attributes
        """
        self.shape = shape or (mu.shape if mu is not None else torch.Size([1]))
        mu = mu if mu is not None else torch.zeros(self.shape)
        std = std if std is not None else torch.ones(self.shape)
        rho = torch.log(torch.exp(std) - 1)  # Inverse of softplus

        if register_fn:
            register_fn("mu", mu)
            register_fn("rho", rho)

        self.mu = mu
        self.rho = rho

    @classmethod
    def from_param(cls, param: Union[torch.nn.Parameter, torch.Tensor]) -> "Gaussian":
        """
        Creates a Gaussian instance from an existing parameter or distribution.

        Args:
            param (Union[torch.nn.Parameter, torch.Tensor]): Source parameter.

        Returns:
            Gaussian: A new Gaussian instance.
        """
        if isinstance(param, (torch.nn.Parameter, torch.Tensor)):
            return cls(mu=param)
        return cls(mu=param.mu, std=param.std)

    @property
    def std(self) -> torch.Tensor:
        """
        Computes the standard deviation using the softplus transformation.

        Returns:
            torch.Tensor: The standard deviation.
        """
        return torch.nn.functional.softplus(self.rho)

    def forward(self, n_samples: int) -> torch.Tensor:
        """
        Samples from the Gaussian distribution.

        Args:
            n_samples (int): The number of samples to generate.

        Returns:
            torch.Tensor: The sampled values.
        """
        epsilon = torch.randn(n_samples, *self.shape)
        return self.mu + epsilon * self.std
