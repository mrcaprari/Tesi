import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from converter_project.algorithms.bbvi import BBVI
from converter_project.algorithms.svgd import SVGD
from converter_project.tests.toy_functions.plot_utilities import *
from converter_project.tests.toy_functions.toy_functions import *

input_shape = 1
output_shape = 1
hidden_shape = 16
hidden_shape_2 = 32
hidden_shape_3 = 64
batch_size = 64
total_data = 64
n_particles = 100
epochs = 100
learning_rate = 0.2
kl_weight = 0.5

dataloader, x_truth, y_truth = create_data_and_ground_truth(
    func=nonlinear_sinusoidal,
    input_shape=input_shape,
    batch_size=batch_size,
    total_data=total_data,
    ground_truth_range=(-3.1, 3.1),
)


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


det_model = SimpleModule()

model, pred_history, kernel_history, total_history = BBVI(
    starting_model=det_model,
    n_samples=n_particles,
    #    n_particles = n_particles,
    epochs=epochs,
    dataloader=dataloader,
    loss_fn=torch.nn.MSELoss(),
    optimizer_fn=torch.optim.Adam,
    learning_rate=learning_rate,
    kl_weight=kl_weight,
)

plot_with_uncertainty_from_dataloader(dataloader, x_truth, y_truth, model, n_particles)


model, pred_history, kernel_history, total_history = SVGD(
    starting_model=det_model,
    #    n_samples=n_particles,
    n_particles=n_particles,
    epochs=epochs,
    dataloader=dataloader,
    loss_fn=torch.nn.MSELoss(),
    optimizer_fn=torch.optim.Adam,
    learning_rate=learning_rate,
    #    kl_weight=kl_weight,
)

plot_with_uncertainty_from_dataloader(dataloader, x_truth, y_truth, model, n_particles)
