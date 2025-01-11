import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from converter_project.converters import ModuleConverter
from converter_project.distributions import Gaussian, Particle, Prior
from converter_project.transformations import (
    GaussianTransformation,
    ParticleTransformation,
)
from converter_project.transformations.method_transformations import (
    initialize_particles,
)


def SVGD(
    starting_model,
    n_samples,
    epochs,
    dataloader,
    loss_fn,
    optimizer_fn,
    learning_rate,
    transform_list=[],
):
    model = ModuleConverter(ParticleTransformation()).convert(
        starting_model, transform_list
    )
    initialize_particles(model, n_samples)

    dataloader_iter = iter(dataloader)
    x_dummy, _ = next(dataloader_iter)  # Peek first batch

    output = model(x_dummy, n_samples)
    optimizer = optimizer_fn(model.parameters(), lr=learning_rate)

    pred_history = []
    kernel_history = []
    total_history = []

    for epoch in range(epochs):
        with tqdm(total=len(dataloader), desc=f"Epoch {epoch + 1}/{epochs}") as pbar:
            for x_batch, y_batch in dataloader:
                pred_loss, kernel_loss = SVGD_step(
                    model, n_samples, x_batch, y_batch, loss_fn, optimizer
                )
                total_loss = pred_loss + kernel_loss

                pred_history.append(pred_loss.detach().cpu().numpy())
                kernel_history.append(kernel_loss.detach().cpu().numpy())
                total_history.append(total_loss.detach().cpu().numpy())

                pbar.set_postfix(
                    tot_loss=total_loss.item(),
                    pred=pred_loss.item(),
                    kernel=kernel_loss.item(),
                )
                pbar.update(1)

    return model, pred_history, kernel_history, total_history


def SVGD_step(model, n_samples, x, y, loss_fn, optimizer):

    optimizer.zero_grad()
    model.compute_kernel_matrix()
    output = model(x, n_samples)
    pred_loss = torch.vmap(loss_fn, in_dims=(0, None))(output, y).mean()
    pred_loss.backward()

    model.perturb_gradients()
    kernel_loss = model.kernel_matrix.sum(dim=1).mean()
    kernel_loss.backward()

    optimizer.step()

    return pred_loss, kernel_loss
