import torch
from stochastic_project.distributions import DistributionFactory, Prior, Particle, Gaussian, RandomParameter

def to_standard_gaussian(param):
    param_prior = DistributionFactory.create(Gaussian, Prior, shape = param.shape)
    random_param = DistributionFactory.create(Gaussian, RandomParameter,prior = param_prior)
    return random_param

def to_gaussian(param):
    param_prior = DistributionFactory.create(Gaussian, Prior, shape = param.shape)
    random_param = DistributionFactory.create(Gaussian, RandomParameter, mu = param,prior = param_prior)
    return random_param

def to_particle(param):
    param_prior = DistributionFactory.create(Gaussian, Prior, shape = param.shape)
    particle_param = Particle(param_prior)
    return particle_param


