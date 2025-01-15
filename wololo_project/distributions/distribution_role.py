from abc import abstractmethod

import torch
from torch.nn.modules.lazy import LazyModuleMixin


class DistributionRole(torch.nn.Module):
    """
    Base class for distribution roles that define specific behaviors to be
    mixed with a distribution family.

    This serves as a modular component to add functionality such as registering
    attributes as trainable parameters or fixed buffers.
    """

    def __init__(self) -> None:
        super().__init__()


class Prior(DistributionRole):
    """
    A distribution role that registers attributes as buffers, representing
    fixed probabilistic definitions.
    """

    def __init__(self) -> None:
        super().__init__()

    def register_fn(self, name: str, attribute: torch.Tensor) -> torch.Tensor:
        """
        Registers an attribute as a buffer.

        Args:
            name (str): The name of the buffer.
            attribute (torch.Tensor): The tensor to be registered as a buffer.

        Returns:
            torch.Tensor: The registered buffer.
        """
        return self.register_buffer(name, attribute)


class RandomParameter(DistributionRole):
    """
    A distribution role that registers attributes as trainable parameters,
    allowing probabilistic parameters to be optimized during training.
    """

    def __init__(self, prior: Prior) -> None:
        super().__init__()
        self.register_module("prior", prior)

    def register_fn(self, name: str, attribute: torch.Tensor) -> torch.nn.Parameter:
        """
        Registers an attribute as a trainable parameter.

        Args:
            name (str): The name of the parameter.
            attribute (torch.Tensor): The tensor to be registered as a parameter. If a scalar value is provided,
                                      its shape will be expanded according to the shape of the RandomParameter.

        Returns:
            torch.nn.Parameter: The registered parameter.
        """
        if attribute.numel() == 1:
            attribute = torch.full(self.shape, attribute.item())
        return self.register_parameter(name, torch.nn.Parameter(attribute))


class Particle(LazyModuleMixin, DistributionRole):
    """
    A distribution role for particle-based representations of a distribution.
    """

    def __init__(self, prior: Prior) -> None:
        """
        Initializes the Particle distribution role.

        This constructor sets up the `Particle` role by registering the prior
        distribution as a module and preparing an uninitialized parameter for particles.
        It also initializes a dictionary to store Einstein summation equations for
        efficient tensor operations.

        Args:
            prior (Prior): The prior distribution to be associated with the particles.
                This defines the distribution from which the particles will be sampled
                and initialized.
        """
        super().__init__()
        self.register_module("prior", prior)
        self.register_parameter("particles", torch.nn.UninitializedParameter())
        self.einsum_equations = {}

    def initialize_parameters(self, n_particles: int) -> None:
        """
        Materializes and initializes particles based on the prior distribution.

        Args:
            n_particles (int): The number of particles to initialize.
        """
        if self.has_uninitialized_params():
            # Materialize the parameter with the correct shape
            self.particles.materialize((n_particles, *self.prior.shape))

        # Initialize particles using the prior's forward method
        with torch.no_grad():
            self.particles = torch.nn.Parameter(self.prior.forward(n_particles))

    @property
    def flattened_particles(self) -> torch.Tensor:
        """
        Returns the particles flattened along all but the first dimension.

        Returns:
            torch.Tensor: Flattened particles.
        """
        return torch.flatten(self.particles, start_dim=1)

    def general_product(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Computes a generalized tensor product using Einstein summation notation.

        Args:
            A (torch.Tensor): The left-hand operand.
            B (torch.Tensor): The right-hand operand.

        Returns:
            torch.Tensor: The result of the tensor product.
        """
        ndim_B = B.ndim
        if ndim_B not in self.einsum_equations:
            suffix = "".join([chr(ord("k") + i) for i in range(ndim_B - 1)])
            self.einsum_equations[ndim_B] = f"ij,j{suffix}->i{suffix}"
        equation = self.einsum_equations[ndim_B]
        return torch.einsum(equation, A, B)

    def perturb_gradients(self, kernel_matrix: torch.Tensor) -> None:
        """
        Perturbs the gradients of the parameter particles using a kernel matrix.

        Args:
            kernel_matrix (torch.Tensor): The kernel matrix to perturb gradients.
        """
        self.particles.grad = self.general_product(kernel_matrix, self.particles.grad)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Forward method to return the particles.

        Returns:
            torch.Tensor: The particles.
        """
        return self.particles
