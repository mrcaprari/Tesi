import torch.fx
import torch
from converter_project.converters import ModuleConverter
from converter_project.transformations import GaussianTransformation


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

transform_list = ['linear','second.bias']
stochastic_model = ModuleConverter(GaussianTransformation()).convert(model, transform_list)

print("Buffers:")
print(list(stochastic_model.named_buffers()))
print("Params:")
print(list(stochastic_model.named_parameters()))

# Run the model
