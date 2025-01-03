# Prendere output da grafo trasformato

# Calcolare loss in parallelo

# Calcolare media loss

# Calcolare KL divergence

# Sommare

# Ottimizzare


# Test
import torch
from torch import nn
from converter_project.transformations import GaussianTransformation
from converter_project.converters import ModuleConverter
from converter_project.distributions import Particle, Gaussian, Prior

input_shape = 2
output_shape = 1
hidden_shape = 4
batch_size = 256
n_particles = 2

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
x = torch.randn(batch_size, input_shape)
y = torch.randn(batch_size, output_shape)

transform_list = ['linear.weight', 'last.bias']
gaussian_model = ModuleConverter(GaussianTransformation()).convert(model, transform_list)


loss_fn = nn.MSELoss()
loss_history = []

for i in range(2000):
    output = gaussian_model(x, n_samples = 500)

    losses = torch.vmap(loss_fn, in_dims=(0, None))(output, y) 
    kl = gaussian_model.kl_divergence()

    total_loss = losses + kl * 0.0005
    print(total_loss.mean())
    loss_history.append(total_loss.mean().detach().numpy())

    total_loss.backward(gradient = torch.ones_like(losses), retain_graph=True)

    optimizer = torch.optim.Adam(gaussian_model.parameters(), lr=0.005)
    optimizer.step()

# plot loss history
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

plt.plot(loss_history)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss History')
plt.show()
#plt.savefig('loss_history.png')  # Save the plot as an image

