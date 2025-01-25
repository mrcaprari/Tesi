import math

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from ..converters import Converter
from .svgd import SVGD

input_shape = 1
output_shape = 1
hidden_shape = 128
hidden_shape_2 = 256
hidden_shape_3 = 512
batch_size = 64
total_data = 32
n_samples = 100
epochs = 200
learning_rate = 0.01


# Ground truth function
def ground_truth(func, precision=500):
    x_range = torch.linspace(start=-5, end=5, steps=precision).unsqueeze(1)
    y_true = func(x_range)
    return x_range, y_true


def create_data(func, input_shape, total_data):
    input_data = torch.rand(total_data, input_shape) * 6 - 3
    target_data = func(input_data)
    return input_data, target_data


def create_dataloader(input_data, target_data, batch_size):
    dataset = TensorDataset(input_data, target_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return input_data, target_data, dataloader


def create_data_and_ground_truth(
    func, input_shape, total_data, batch_size, ground_truth_range=(-5, 5), precision=500
):
    # Generate random input data and target data
    input_data = torch.randn(total_data, input_shape)
    target_data = func(input_data)

    # Create DataLoader
    dataset = TensorDataset(input_data, target_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Compute ground truth
    x_range = torch.linspace(
        ground_truth_range[0], ground_truth_range[1], precision
    ).unsqueeze(1)
    y_true = func(x_range)

    return dataloader, x_range, y_true


def nonlinear_sinusoidal(x):
    nonlinear_output = (
        (torch.sin(x * 8) * 3 + x - torch.exp(x) * 0.2) * torch.exp(-x.pow(2)) * 3
    )  # Apply sine transformation
    return nonlinear_output


class ParticleParam(torch.nn.modules.lazy.LazyModuleMixin, torch.nn.Module):
    """
    A module to represent stochastic parameter particles for particle-based inference.

    Args:
        parameter (torch.Tensor): The parameter to be modeled with particles.
    """

    def __init__(self, parameter):
        super().__init__()
        self.prior = torch.distributions.Normal(
            loc=parameter, scale=torch.full(parameter.size(), 1.0)
        )
        self.register_parameter("particles", torch.nn.UninitializedParameter())
        self.einsum_equations = {}

    def initialize_parameters(self, n_samples: int) -> None:
        if self.has_uninitialized_params():
            self.particles.materialize((n_samples, *self.prior.loc.shape))

        with torch.no_grad():
            self.particles = torch.nn.Parameter(
                self.prior.rsample(
                    (n_samples,),
                )
            )

    @property
    def flattened_particles(self) -> torch.Tensor:
        return torch.flatten(self.particles, start_dim=1)

    def general_product(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        ndim_B = B.ndim
        if ndim_B not in self.einsum_equations:
            suffix = "".join([chr(ord("k") + i) for i in range(ndim_B - 1)])
            self.einsum_equations[ndim_B] = f"ij,j{suffix}->i{suffix}"
        equation = self.einsum_equations[ndim_B]
        return torch.einsum(equation, A, B)

    def perturb_gradients(self, kernel_matrix: torch.Tensor) -> None:
        """
        Modifies particle gradients multiplying them by the kernel matrix
        to reflect particle interactions.

        Args:
            kernel_matrix (torch.Tensor): Kernel matrix encoding pairwise distances.
        """
        self.particles.grad = self.general_product(kernel_matrix, self.particles.grad)

    def forward(self, n_samples):
        """
        Forward pass returning the particles.

        Args:
            n_samples (int): Number of particle realizations.

        Returns:
            torch.Tensor: The particles.
        """
        return self.particles


class SimpleModule(torch.nn.Module):
    def __init__(self, input_size, hidden_size, hidden_size_2, output_size):
        super().__init__()
        self.first = torch.nn.Linear(input_size, hidden_size)
        self.second = torch.nn.Linear(hidden_size, hidden_size_2)
        self.third = torch.nn.Linear(hidden_size_2, output_size)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.first(x)
        x = self.relu(x)
        x = self.second(x)
        x = self.relu(x)
        x = self.third(x)
        return x


fake_input = torch.randn(batch_size, input_shape)
fake_output = torch.randn(batch_size, output_shape)


# Example usage
# Define a fully connected model with input=5, two hidden layers of size 3, and output=2
det_model = SimpleModule(input_shape, hidden_shape, hidden_shape_2, output_shape)

dataloader, x_truth, y_truth = create_data_and_ground_truth(
    func=nonlinear_sinusoidal,
    input_shape=input_shape,
    batch_size=batch_size,
    total_data=total_data,
    ground_truth_range=(-3.1, 3.1),
)

stoch_model, pred_history, kl_history, total_history = SVGD(
    starting_model=det_model,
    n_samples=n_samples,
    epochs=epochs,
    dataloader=dataloader,
    loss_fn=torch.nn.MSELoss(),
    optimizer_fn=torch.optim.Adam,
    learning_rate=learning_rate,
)


def get_mixed_dict(original_dict, mixed_dict, layer_names, unit_index):
    # Make a copy of the permuted_dict to avoid modifying the original input
    first_layer, second_layer = layer_names

    # Update the required entries
    mixed_dict.get(f"{first_layer}.weight.particles")[:, unit_index, :] = (
        original_dict.get(f"{first_layer}.weight.particles")[:, unit_index, :]
    )

    mixed_dict.get(f"{first_layer}.bias.particles")[:, unit_index] = original_dict.get(
        f"{first_layer}.bias.particles"
    )[:, unit_index]

    mixed_dict.get(f"{second_layer}.weight.particles")[:, :, unit_index] = (
        original_dict.get(f"{second_layer}.weight.particles")[:, :, unit_index]
    )

    return mixed_dict


def get_layer_pairs_with_params(model):
    # Get all layers with parameters
    layers_with_params = [
        (name, layer)
        for name, layer in model.named_children()
        if len(list(layer.parameters())) > 0
    ]

    # Create pairs of subsequent layers
    pairs = [
        (layers_with_params[i], layers_with_params[i + 1])
        for i in range(len(layers_with_params) - 1)
    ]

    return pairs


def sobol_sensitivity_analysis(
    stochastic_model, deterministic_model, fake_input, n_samples
):
    # Step 1: Get parameter dictionaries
    parameters_dict = {
        name: param.clone() for name, param in stochastic_model.named_parameters()
    }

    permutation = torch.randperm(n_samples)
    permutation_dict = {
        name: param[permutation].clone()
        for name, param in stochastic_model.named_parameters()
    }

    mixed_dict = {
        name: param[permutation].clone()
        for name, param in stochastic_model.named_parameters()
    }

    # Step 2: Compute outputs for non-permuted and fully-permuted parameters
    y_A = torch.func.functional_call(
        stochastic_model, parameters_dict, (fake_input, n_samples)
    ).squeeze(1)

    y_A_mean = y_A.mean()

    y_B = torch.func.functional_call(
        stochastic_model, permutation_dict, (fake_input, n_samples)
    ).squeeze(1)

    print(y_A.shape)
    print(y_B.shape)

    denom = torch.dot(y_A, y_A) - y_A_mean.pow(2)

    sensitivity_results = []

    # Step 3: Loop through layer pairs
    for layer_pair in get_layer_pairs_with_params(deterministic_model):
        (first_name, first_layer), (second_name, second_layer) = layer_pair
        num_output_units = first_layer.out_features

        # Step 4: Loop through each output unit
        for output_unit in range(num_output_units):
            # Compute mixed parameters for the current unit
            mixed_parameters = get_mixed_dict(
                parameters_dict,
                mixed_dict,
                layer_names=(first_name, second_name),
                unit_index=output_unit,
            )

            y_Ci = torch.func.functional_call(
                stochastic_model, mixed_parameters, (fake_input, n_samples)
            ).squeeze(1)

            # Step 5: Compute Sobol indices

            Si = (torch.dot(y_A, y_Ci) - y_A_mean.pow(2)) / denom
            STi = 1 - (torch.dot(y_B, y_Ci) - y_A_mean.pow(2)) / denom

            sensitivity_results.append(
                {"unit": output_unit, "Si": Si.item(), "STi": STi.item()}
            )

    return sensitivity_results


results = sobol_sensitivity_analysis(
    stochastic_model=stoch_model,
    deterministic_model=det_model,
    fake_input=torch.ones(input_shape),
    n_samples=n_samples,
)

total_S_Ti = 0
total_Si = 0


for result in results:
    total_Si += result["Si"]
    total_S_Ti += result["STi"]
    print(f"Unit: {result['unit']}, S_i: {result['Si']:.4f}, S_Ti: {result['STi']:.4f}")

print(total_Si)
print(total_S_Ti)
