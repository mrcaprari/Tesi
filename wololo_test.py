import torch
from torch import nn
from tqdm import tqdm

from tests.toy_functions.plot_utilities import *
from tests.toy_functions.toy_functions import *
from wololo import BBVIConverter, SVGDConverter
from wololo.algorithms import BBVI, SVGD

input_shape = 1
output_shape = 1
hidden_shape = 16
hidden_shape_2 = 64
hidden_shape_3 = 32
batch_size = 64
total_data = 32
n_samples = 50
epochs = 500
learning_rate = 0.005


class SimpleModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(input_shape, hidden_shape)
        self.middle = nn.Linear(hidden_shape, hidden_shape_2)
        self.middle2 = nn.Linear(hidden_shape_2, hidden_shape_3)
        self.last = nn.Linear(hidden_shape_3, output_shape)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        x = self.middle(x)
        x = self.relu(x)
        x = self.middle2(x)
        x = self.relu(x)
        x = self.last(x)
        return x


dataloader, x_truth, y_truth = create_data_and_ground_truth(
    func=nonlinear_sinusoidal,
    input_shape=input_shape,
    batch_size=batch_size,
    total_data=total_data,
    ground_truth_range=(-3.1, 3.1),
)


class GaussianParameter(torch.nn.Module):
    def __init__(self, parameter):
        super().__init__()
        self.mu = torch.nn.Parameter(parameter)
        self.rho = torch.nn.Parameter(
            torch.full_like(parameter, torch.log(torch.exp(torch.tensor(0.01)) - 1.0))
        )
        self.prior_mu = torch.full_like(parameter, 0.0)
        self.prior_std = torch.full_like(parameter, 1.0)

    @property
    def std(self):
        return torch.nn.functional.softplus(self.rho)

    @property
    def prior(self):
        return torch.distributions.Normal(self.prior_mu, self.prior_std)

    @property
    def dist(self):
        return torch.distributions.Normal(self.mu, self.std)

    def forward(self, n_samples):
        return self.dist.rsample((n_samples,))  # Reparameterized sampling

    def kl_divergence(self):
        return torch.distributions.kl_divergence(self.dist, self.prior).sum()


BBVI_model = BBVIConverter().convert(SimpleModule(), GaussianParameter)

pred_history, kl_history, total_history = BBVI(
    BBVI_model=BBVI_model,
    n_samples=n_samples,
    epochs=epochs,
    train_loader=dataloader,
    loss_fn=torch.nn.MSELoss(),
    optimizer_fn=torch.optim.Adam,
    learning_rate=learning_rate,
)

plot_with_uncertainty_from_dataloader(
    dataloader, x_truth, y_truth, BBVI_model, n_samples
)


fake_input = torch.randn(1, 1)

SVGD_model = SVGDConverter().convert(
    SimpleModule(), particle_config={"prior_std": 0.001}
)

SVGD_model(fake_input, n_samples)

pred_history, kl_history, total_history = SVGD(
    SVGD_model=SVGD_model,
    n_samples=n_samples,
    epochs=epochs,
    train_loader=dataloader,
    loss_fn=torch.nn.MSELoss(),
    optimizer_fn=torch.optim.Adam,
    learning_rate=learning_rate,
)


plot_with_uncertainty_from_dataloader(
    dataloader, x_truth, y_truth, SVGD_model, n_samples
)
