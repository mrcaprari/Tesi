import torch


class ParticleParameter(torch.nn.modules.lazy.LazyModuleMixin, torch.nn.Module):
    def __init__(self, parameter, particle_config: dict = None):
        """
        Args:
            parameter (torch.Tensor): The tensor parameter to be wrapped.
            particle_config (dict, optional): Configuration options for the particle.
                Defaults to {"prior_std": 1.0}. If the key "prior" is provided, that
                distribution will be used instead of constructing a Normal distribution.
        """
        super().__init__()

        self.particle_config = particle_config or {"prior_std": 1.0}

        if "prior" in self.particle_config:
            self.prior = self.particle_config["prior"]
        else:
            self.prior = torch.distributions.Normal(
                loc=parameter,
                scale=torch.full(parameter.size(), self.particle_config["prior_std"]),
            )

        self.register_parameter("particles", torch.nn.UninitializedParameter())

    def initialize_parameters(self, n_samples: int) -> None:
        """
        Materializes and initializes the particle parameters for sampling.

        Args:
            n_samples (int): Number of samples/particles to generate.
        """
        if n_samples <= 0:
            raise ValueError("n_samples must be a positive integer.")

        if self.has_uninitialized_params():
            self.particles.materialize((n_samples, *self.prior.loc.shape))

        with torch.no_grad():
            self.particles = torch.nn.Parameter(
                self.prior.rsample(
                    (n_samples,),
                )
            )

        # Precompute einsum (parameters can have different dimensions, e.g. weight vs bias)
        suffix = "".join(
            [chr(ord("k") + i) for i in range(self.particles.ndim - 1)]
        )  # Dynamically compute suffix
        self.einsum_equation = f"ij,j{suffix}->i{suffix}"  # Precompute einsum equation

    @property
    def flattened_particles(self) -> torch.Tensor:
        return torch.flatten(self.particles, start_dim=1)

    def perturb_gradients(self, kernel_matrix: torch.Tensor) -> None:
        # Direct einsum operation using the precomputed einsum equation
        self.particles.grad = torch.einsum(
            self.einsum_equation, kernel_matrix, self.particles.grad
        )

    def forward(self, n_samples):
        return self.particles


def svgd_initialize(module) -> None:
    """
    Initializes the BBVI-specific attributes for the module:
    - Prepares a list of stochastic parameters.
    - Computes the KL divergence normalization denominator.
    """
    # Prepare list of stochastic parameters
    module.particle_modules = module.particle_modules = [
        pm for pm in module.modules() if isinstance(pm, ParticleParameter)
    ]


def all_particles(module: torch.nn.Module) -> torch.Tensor:
    """
    Concatenate all particles into a single tensor.

    Returns:
        torch.Tensor: Flattened and concatenated particles from all submodules.
    """
    return torch.cat(
        [
            torch.flatten(particle_mod.particles, start_dim=1)
            for particle_mod in module.particle_modules
        ],
        dim=1,
    )


def compute_kernel_matrix(module: torch.nn.Module) -> None:
    """
    Computes the RBF kernel matrix for the particles in the model.
    """
    # Gather all flattened particles
    particles = module.all_particles()
    n_particles = particles.shape[0]

    # Compute lengthscale based on median pairwise squared distances
    pairwise_sq_dists = torch.cdist(particles, particles, p=2) ** 2
    median_squared_dist = pairwise_sq_dists.median()
    lengthscale = torch.sqrt(
        0.5
        * median_squared_dist
        / (torch.log(torch.tensor(n_particles, dtype=particles.dtype)) + 1e-8)
    )

    # Compute kernel matrix
    module.kernel_matrix = torch.exp(-pairwise_sq_dists / (2 * lengthscale**2 + 1e-8))


def perturb_gradients(module: torch.nn.Module) -> None:
    """
    Adjust gradients of all particles in the model using the kernel matrix.
    """
    module.compute_kernel_matrix()
    for particle in module.particle_modules:
        particle.perturb_gradients(module.kernel_matrix)
