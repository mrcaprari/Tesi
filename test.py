import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader, TensorDataset

from wololo import BBVIConverter, SVGDConverter

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


def nonlinear_function(x):
    """A non-linear function combining a sine term (with scaling) and a quadratic term."""
    return torch.sin(x * 2) * 4 + 0.1 * x**2


def generate_data(num_train):
    # Create a fine grid for ground truth
    x_fine = torch.linspace(-4.51, 4.51, steps=1000).unsqueeze(1)  # Shape: [100, 1]
    y_true = nonlinear_function(x_fine)

    # Create training dataset by selecting 50 random points and adding noise
    indices = torch.randperm(x_fine.size(0))[:num_train]
    x_train = x_fine[indices]
    y_train_clean = nonlinear_function(x_train)
    noise = torch.randn_like(y_train_clean) * 0.05
    y_train = y_train_clean + noise

    return x_fine, y_true, x_train, y_train


class SimpleModule(torch.nn.Module):
    def __init__(self, hidden_shape=16, hidden_shape_2=32):
        super().__init__()
        self.fc1 = torch.nn.Linear(1, hidden_shape)
        self.fc2 = torch.nn.Linear(hidden_shape, hidden_shape_2)
        self.fc3 = torch.nn.Linear(hidden_shape_2, 1)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


class GaussianParameter(torch.nn.Module):
    def __init__(self, parameter):
        super().__init__()
        self.mu = torch.nn.Parameter(parameter)
        self.rho = torch.nn.Parameter(
            torch.full_like(parameter, torch.log(torch.expm1(torch.tensor(0.01))))
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
        return torch.distributions.kl_divergence(self.dist, self.prior).mean()


def train_SVGD(
    epochs, learning_rate, n_samples, hidden_shape, hidden_shape_2, dataloader
):

    # Convert to non-parametric BNN suitable for SVGD
    model = SVGDConverter().convert(
        SimpleModule(hidden_shape, hidden_shape_2), particle_config={"prior_std": 1.0}
    )
    dummy_input = torch.randn(1, 1)
    model(dummy_input, n_samples)

    # Define loss and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    vmap_loss = torch.vmap(criterion, in_dims=(0, None))

    for epoch in range(epochs):
        batch_bar = tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        epoch_pred_losses = []
        epoch_kernel_losses = []

        for x_batch, y_batch in batch_bar:
            optimizer.zero_grad()

            predictions = model(
                x_batch, n_samples
            )  # [n_samples, batch_size, output_dim]
            loss = vmap_loss(predictions, y_batch).mean()
            # Compute standard gradients
            loss.backward()

            # Compute kernel matrix and perturb gradients
            model.perturb_gradients()

            # Compute kernel gradients
            kernel_loss = model.kernel_matrix.sum(dim=1).mean()

            # Accumulate gradients
            kernel_loss.backward()

            optimizer.step()

            epoch_pred_losses.append(loss.item())
            epoch_kernel_losses.append(kernel_loss.item())

            batch_bar.set_postfix(
                {
                    "pred_loss": f"{loss.item():.4f}",
                    "kernel_loss": f"{kernel_loss.item():.4f}",
                }
            )

        if (epoch + 1) % 50 == 0:
            avg_pred = np.mean(epoch_pred_losses)
            avg_kernel = np.mean(epoch_kernel_losses)
            print(
                f"Epoch {epoch+1}: Avg Prediction Loss = {avg_pred:.4f}, Avg Kernel Loss = {avg_kernel:.4f}"
            )

    return model


def train_BBVI(
    epochs, learning_rate, n_samples, hidden_shape, hidden_shape_2, dataloader
):
    # Convert to parametric BNN suitable for BBVI
    model = BBVIConverter().convert(
        SimpleModule(hidden_shape, hidden_shape_2), GaussianParameter
    )

    # Define loss and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    vmap_loss = torch.vmap(criterion, in_dims=(0, None))

    for epoch in range(epochs):
        batch_bar = tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        epoch_pred_losses = []
        epoch_kl_losses = []

        for x_batch, y_batch in batch_bar:
            optimizer.zero_grad()

            predictions = model(
                x_batch, n_samples
            )  # [n_samples, batch_size, output_dim]
            loss = vmap_loss(predictions, y_batch).mean()

            # Compute KL divergence loss
            # KL weight computed to account for mini_batch size and number of variational parameters
            kl_weight = x_batch.size(0) / model.kl_denominator
            kl_loss = model.kl_divergence() * kl_weight

            total_loss = loss + kl_loss
            total_loss.backward()
            optimizer.step()

            epoch_pred_losses.append(loss.item())
            epoch_kl_losses.append(kl_loss.item())

            batch_bar.set_postfix(
                {
                    "pred_loss": f"{loss.item():.4f}",
                    "kernel_loss": f"{kl_loss.item():.4f}",
                }
            )

        if (epoch + 1) % 50 == 0:
            avg_pred = np.mean(epoch_pred_losses)
            avg_kl = np.mean(epoch_kl_losses)
            print(
                f"Epoch {epoch+1}: Avg Prediction Loss = {avg_pred:.4f}, Avg KL Loss = {avg_kl:.4f}"
            )

    return model


def plot_results(model, x_fine, y_true, dataloader, n_samples):
    plt.figure(figsize=(10, 6))

    # Plot training data (scatter)
    for batch_inputs, batch_targets in dataloader:
        plt.scatter(
            batch_inputs.numpy(), batch_targets.numpy(), color="blue", alpha=0.6
        )

    # Plot the ground truth function (fine grid)
    plt.plot(
        x_fine.numpy(), y_true.numpy(), color="green", label="Ground Truth", linewidth=2
    )

    # Obtain model predictions (sample multiple particles for uncertainty)
    with torch.no_grad():
        y_samples = model(x_fine, n_samples)  # [n_samples, num_points, output_dim]
        y_mean = y_samples.mean(dim=0).squeeze().numpy()
        y_std = y_samples.std(dim=0).squeeze().numpy()

    # Plot mean predictions and uncertainty bounds
    plt.plot(
        x_fine.numpy(),
        y_mean,
        color="red",
        label="Model Predictions (Mean)",
        linewidth=2,
    )
    for i in range(1, 4):
        plt.fill_between(
            x_fine.squeeze().numpy(),
            y_mean - i * y_std,
            y_mean + i * y_std,
            alpha=0.2 / i,
            label=f"{i}-σ Bound" if i == 1 else None,
        )

    plt.xlim([-4.5, 4.5])
    plt.ylim([-6, 6])
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.legend()
    plt.title("Ground Truth, Training Data, and Model Predictions with Uncertainty")
    plt.grid(True)
    plt.show()


def main():
    # Parameterization via argparse
    parser = argparse.ArgumentParser(
        description="Test Wololo Bayesian Neural Network on a toy non-linear function."
    )
    parser.add_argument(
        "--epochs", type=int, default=600, help="Number of training epochs."
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.005, help="Learning rate."
    )
    parser.add_argument(
        "--n_samples", type=int, default=5, help="Number of particles for predictions."
    )
    parser.add_argument(
        "--num_train", type=int, default=50, help="Number of particles for predictions."
    )

    parser.add_argument(
        "--hidden_shape", type=int, default=16, help="Hidden layer size (first layer)."
    )
    parser.add_argument(
        "--hidden_shape_2",
        type=int,
        default=32,
        help="Hidden layer size (second layer).",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for training."
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["svgd", "bbvi"],
        default="svgd",
        help="Algorithm to use: 'svgd' (default) or 'bbvi'.",
    )
    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(42)
    np.random.seed(42)

    # Generate data
    x_fine, y_true, x_train, y_train = generate_data(args.num_train)
    dataset = TensorDataset(x_train, y_train)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Choose training method based on the algorithm argument
    if args.algorithm.lower() == "bbvi":
        model = train_BBVI(
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            n_samples=args.n_samples,
            hidden_shape=args.hidden_shape,
            hidden_shape_2=args.hidden_shape_2,
            dataloader=dataloader,
        )
    else:
        model = train_SVGD(
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            n_samples=args.n_samples,
            hidden_shape=args.hidden_shape,
            hidden_shape_2=args.hidden_shape_2,
            dataloader=dataloader,
        )

    plot_results(model, x_fine, y_true, dataloader, args.n_samples)


if __name__ == "__main__":
    main()
