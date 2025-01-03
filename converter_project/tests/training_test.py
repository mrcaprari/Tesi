import torch
from torch import nn
from converter_project.transformations import GaussianTransformation, ParticleTransformation
from converter_project.converters import ModuleConverter
from converter_project.distributions import Particle, Gaussian, Prior
from torch.utils.data import DataLoader, TensorDataset
from converter_project.algorithms.svgd import SVGD

input_shape = 784
output_shape = 10
hidden_shape = 64
batch_size = 256
n_particles = 150


def create_dataloader(batch_size, total_data):
    input_data = torch.randn(total_data, input_shape)  
    target_data = torch.sin(input_data)@torch.randn(input_shape, output_shape) 
    dataset = TensorDataset(input_data, target_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

class SimpleModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(input_shape, hidden_shape)
        self.relu = nn.ReLU()
        self.last = nn.Linear(hidden_shape, output_shape)

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        x = self.last(x)
        return x

model = SimpleModule()
dataloader = create_dataloader(2000, 256)

loss_history = SVGD(
    starting_model=model,
    n_particles = n_particles,
    epochs = 55,
    dataloader=dataloader,
    loss_fn = nn.MSELoss(),
    optimizer_fn = torch.optim.Adam, 
    learning_rate = 0.01
)

#plot loss history
import matplotlib.pyplot as plt
plt.plot(loss_history)
plt.show()