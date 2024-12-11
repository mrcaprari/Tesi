import torch
import stochastic_project

in_features = 784
out_features = 10
hidden_shape = 64
batch_size = 256
x = torch.randn(batch_size, in_features)

class BaseModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(in_features, hidden_shape)
        self.linear2 = torch.nn.Linear(hidden_shape, out_features)

    def forward(self, x):
        x = self.linear1(x)
        return self.linear2(x)

# Already existing module 
prova = BaseModel()

stochastic_project.utils.convert_to_stochastic_parameters(prova.linear1)

# tracer = stochastic_project.custom_tracers.NoLeafTracer()
# gf = stochastic_project.graph_factory.GraphFactory(prova, tracer)
# print(gf.deterministic_graph())

# Prepare transformations 
#   original_graph = tracer.trace(original_module)
#   utils.convert_to_stochastic_parameters(original_module)

### node manipulations
#       create_deterministic_graph()
#           inizializziamo det_graph e det_val_map
#           sostituiamo nodi
#           restituiamo torch.fx.GraphModule(original_module, deterministic_graph) 
#           -> Pronti per chiamare modello

#       create_stochastic_graph()
#           inizializziamo stoch_graph e stoch_val_map
#           sostituiamo e aggiungiamo nodi
#           restituiamo torch.fx.GraphModule(original_module, stochastic_graph)
#           lo passiamo dentro NewVmapApplicator
#           applichiamo transform() method
#           -> Pronti per chiamare modello

#       
### NewVmapApplicator
#   wrap fx_builder
#   modify run_node
#   modify call_function

