from typing import Any

import torch

from converter_project.distributions import (
    DistributionFactory,
    Gaussian,
    Particle,
    Prior,
    RandomParameter,
)


def to_standard_gaussian(param: Any) -> RandomParameter:
    """
    Transforms a parameter into a random Gaussian parameter with a standard Gaussian prior.

    Args:
        param (Any): The input parameter to transform.

    Returns:
        RandomParameter: The transformed parameter with a standard Gaussian prior.
    """
    param_prior = DistributionFactory.create(Gaussian, Prior, shape=param.shape)
    random_param = DistributionFactory.create(
        Gaussian, RandomParameter, prior=param_prior
    )
    return random_param


def to_gaussian(param: Any) -> RandomParameter:
    """
    Transforms a parameter into a random Gaussian parameter with its mean set to the input parameter.

    Args:
        param (Any): The input parameter to transform.

    Returns:
        RandomParameter: The transformed parameter with a Gaussian prior centered at `param`.
    """
    param_prior = DistributionFactory.create(Gaussian, Prior, shape=param.shape)
    random_param = DistributionFactory.create(
        Gaussian, RandomParameter, mu=param, prior=param_prior, std=torch.tensor(0.01)
    )
    return random_param


def to_particle(param: Any) -> Particle:
    """
    Transforms a parameter into a particle-based parameter with a Gaussian prior.

    Args:
        param (Any): The input parameter to transform.

    Returns:
        Particle: The particle representation of the parameter.
    """
    param_prior = DistributionFactory.create(Gaussian, Prior, shape=param.shape)
    particle_param = Particle(param_prior)
    return particle_param
