import torch
import matplotlib.pyplot as plt

def plot_with_uncertainty_from_dataloader(dataloader, x_range, y_true, model, n_particles, resolution=500):
    plt.figure(figsize=(10, 6))
    
    # Scatter plot of the dataset from DataLoader
    for batch_inputs, batch_targets in dataloader:
        plt.scatter(batch_inputs.numpy(), batch_targets.numpy(), color='blue', alpha=0.6)
    
    # Plot the ground truth function
    plt.plot(x_range.numpy(), y_true.numpy(), color='green', label='Ground Truth', linewidth=2)
    
    # Predict using the model
    with torch.no_grad():
        y_samples = model(x_range, n_particles)  # Sampling predictions
        y_mean = y_samples.mean(dim=0).squeeze().numpy()
        y_std = y_samples.std(dim=0).squeeze().numpy()
    
    # Plot mean predictions
    plt.plot(x_range.numpy(), y_mean, color='red', label='Model Predictions (Mean)', linewidth=2)
    
    # Plot uncertainty bounds
    for i in range(1, 4):  # 1st, 2nd, and 3rd standard deviations
        plt.fill_between(
            x_range.squeeze().numpy(),
            y_mean - i * y_std,
            y_mean + i * y_std,
            alpha=0.2 / i,  # Decreasing alpha for higher deviations
            label=f'{i}-σ Bound'
        )
    
    # Labels and legend
    plt.xlabel('Input')
    plt.ylabel('Output')
    plt.legend()
    plt.title('Ground Truth, Dataset, Model Predictions, and Uncertainty')
    plt.grid()
    plt.show()
