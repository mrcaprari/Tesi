import torch
import stochastic_project

in_features = 10
out_features= 2
batch_size = 5
x = torch.randn(batch_size, in_features)
linear = torch.nn.Linear(in_features, out_features)
output = linear(x)

stoch_linear = stochastic_project.graph_manipulation.create_stochastic_graph(linear)

# stochastic_branch
vmap_transformer = stochastic_project.vmap_interprerer.NewVmapApplicator(stoch_linear)

transformed_model = vmap_transformer.transform()
print(transformed_model.graph)

output = transformed_model(x, n_samples =10)
print(output.shape)

# output = stoch_module.run(x, 50)
# print(output, output.shape)

# print(points_to_stochastic_parameters(linear, 'weight'))

# x = torch.randn(batch_size, in_features)

# output = linear(x)
# print(output)

# det_module = create_deterministic_graph(linear)

# print(det_module(x))
