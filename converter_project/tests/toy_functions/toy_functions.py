import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Ground truth function
def ground_truth(func, precision=500):
    x_range = torch.linspace(start=-5, end=5, steps=precision).unsqueeze(1)
    y_true = func(x_range)
    return x_range, y_true

def create_data(func, input_shape, total_data):
    input_data = torch.rand(total_data, input_shape)*16-8
    target_data = func(input_data)
    return input_data, target_data

def create_dataloader(input_data, target_data, batch_size):
    dataset = TensorDataset(input_data, target_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return input_data, target_data, dataloader

def create_data_and_ground_truth(func, input_shape, total_data, batch_size, ground_truth_range=(-5, 5), precision=500):
    # Generate random input data and target data
    input_data = torch.randn(total_data, input_shape)
    target_data = func(input_data)
    
    # Create DataLoader
    dataset = TensorDataset(input_data, target_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Compute ground truth
    x_range = torch.linspace(ground_truth_range[0], ground_truth_range[1], precision).unsqueeze(1)
    y_true = func(x_range)
    
    return dataloader, x_range, y_true


def nonlinear_sinusoidal(x):
    nonlinear_output = (torch.sin(x * 8)*3 + x - torch.exp(x)*0.2)*torch.exp(-x.pow(2))*3 # Apply sine transformation
    return nonlinear_output