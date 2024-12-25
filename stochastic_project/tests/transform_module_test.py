import torch.fx
import torch
from stochastic_project.module_converter import ModuleConverter
from stochastic_project.transformations import GaussianTransformation


class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10,5)
        self.second = torch.nn.Linear(5,1)
    def forward(self,x):
        return self.second(self.linear(x))

# Instantiate and trace the module
#simple_module = SimpleModule()
model = MyModel()
x = torch.randn(25, 10)

transform_list = ['linear','second.bias']
stochastic_model = ModuleConverter(GaussianTransformation()).convert(model, transform_list)

print("Buffers:")
print(list(stochastic_model.named_buffers()))
print("Params:")
print(list(stochastic_model.named_parameters()))

# for name, param in trans_graph.named_parameters():
#     print(name, param)

# ###### Trainiamolo #####
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset

# def create_toy_dataset(num_samples=100, input_dim=10, seed=42):
#     torch.manual_seed(seed)
#     x = torch.randn(num_samples, input_dim)  # Random input tensor
#     y = x.sum(dim=1, keepdim=True) + torch.randn(num_samples, 1) * 0.1  # Target: sum of inputs + noise
#     return x, y

# batch_size = 8
# num_epochs = 20
# learning_rate = 0.01

# # Create dataset and DataLoader
# x, y = create_toy_dataset(num_samples=100)
# dataset = TensorDataset(x, y)
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# criterion = nn.MSELoss()  # Mean Squared Error loss for regression
# optimizer = optim.Adam(trans_graph.parameters(), lr=learning_rate)

# for epoch in range(num_epochs):
#     for batch_idx, (inputs, targets) in enumerate(dataloader):
#         # Forward pass
#         outputs = trans_graph(inputs, n_samples = 50)
#         mean_outputs = torch.mean(outputs,0)
#         loss = criterion(mean_outputs, targets)

#         # Backward pass and optimization
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#     print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")


# for name, param in trans_graph.named_parameters():
#     print(name, param)



# print(graph)
# print("\n")
# for node in graph.nodes:
#     print(node.name, node.meta)
