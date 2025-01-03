import torch
from torch import nn
from converter_project.transformations import GaussianTransformation, ParticleTransformation
from converter_project.converters import ModuleConverter
from converter_project.distributions import Particle, Gaussian, Prior
from torch.utils.data import DataLoader, TensorDataset


def SVGD(starting_model, n_particles, epochs, dataloader, loss_fn, optimizer_fn, learning_rate, transform_list = []):
    model = ModuleConverter(ParticleTransformation()).convert(starting_model, transform_list)

    model.n_particles = n_particles
    loss_history = []
    for epoch in range(epochs):
        for x_batch, y_batch in dataloader:
            loss = SVGD_step(model, n_particles, x_batch, y_batch, loss_fn, optimizer_fn, learning_rate)
            loss_history.append(loss.mean().detach().cpu().numpy())
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.mean()}")

    return loss_history


def SVGD_step(model, n_particles, x, y, loss_fn, optimizer_fn, learning_rate):
    output = model(x, n_particles)

    optimizer = optimizer_fn(model.parameters(), lr=learning_rate)
    optimizer.zero_grad()    

    
    model.compute_kernel_matrix()

    losses = torch.vmap(loss_fn, in_dims=(0, None))(output, y)
    losses.backward(gradient = torch.ones_like(losses))

    model.perturb_gradients()
    model.kernel_matrix.sum(dim=1).backward(gradient = torch.ones(n_particles))

    optimizer.step()

    return losses


