# Wololo: A PyTorch Framework for Probabilistic Model Conversion

Wololo is a lightweight framework that transforms deterministic PyTorch models into Bayesian models. Using state-of-the-art techniques like Black-Box Variational Inference (BBVI) and Stein Variational Gradient Descent (SVGD), Wololo replaces deterministic parameters with stochastic counterparts.

## Features

- **Model Conversion:** Convert any fully connected PyTorch neural network into a Bayesian Neural Network (BNN) using either BBVI or SVGD.
- **BBVI and SVGD Algorithms:** Choose between BBVI (parametric) and SVGD (non-parametric) conversion methods via a simple command-line argument.
- **PyTorch Integration:** Designed to integrate with existing PyTorch models: simply pass your model to the converter and start experimenting with Bayesian inference
- **Customizable Stochastic Parameters:** Define your own stochastic parameter logic for BBVI or use built-in particle configurations for SVGD.
- **Hybrid Bayesian Models:** Mix deterministic and stochastic layers within the same model.

## Installation
Wololo can be installed via PyPI:
### Via PyPI

```bash
pip install wololo
```
To run the test smoothly, install with optional dependencies:
```bash
pip install wololo[test]
```
This will also install `matplotlib`, `numpy` and `tqdm`.

## Dependencies
Wololo requires:
- `PyTorch` >=2.5.1
- Optional dependencies (for the test): `numpy`, `matplotlib`, `tqdm`

## Usage
Wololo provides two main conversion routines:
**BBVI Conversion** 
For BBVI, the stochastic parameter logic needs to be defined inside a torch.nn.Module. For example:
```python 
class GaussianParameter(torch.nn.Module):
    def __init__(self, parameter):
        super().__init__()
        self.mu = torch.nn.Parameter(parameter)
        self.rho = torch.nn.Parameter(
            torch.full_like(parameter, torch.log(torch.expm1(torch.tensor(0.01)))) #Ensure positivity with rho reparameterization
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
```
This stochastic parameter must implement a `forward()` method responsible for sampling, a `kl_divergence()` method for computing the KL divergence for the prior and its initialized needs the original `torch.nn.Parameter`.
The BBVIConverter() will take care of the rest:
```python 
BBVI_model = BBVIConverter().convert(MyModel(), GaussianParameter)
```

**SVGD Conversion**
For SVGD, just pass pass it to the SVGDConverter:
```python 
SVGD_model = SVGDConverter().convert(MyModel())
```
Then, to initialize the particles perform a dummy run with the specied number of particles:
```python 
n_samples = 10
dummy_input = torch.randn(batch_size, input_dim)
SVGD_model(dummy_input, n_samples)
```
The model is not ready for training and inference with probabilistic predictions.
## Training and Inference
The converted models will now output probabilistic predictions. This needs to be taken into account when performing inference. We suggest using `torch.vmap` to vectorize the computation of the loss:
```python
probabilistic_preds = Wololo_model(input, n_samples) # Shape: [n_samples, batch_size, output_dim]
criterion = torch.nn.MSELoss()
vmap_criterion = torch.vmap(criterion, in_dims = (0, None))
# Compute mean loss
prediction_loss = vmap_criterion(probabilistic_preds, output).mean()
```
Finally, to align with the variational posterior approximation algorithms, the backpropagation step needs to be modified.
For BBVI:
```python
### BBVI
# Compute mean loss as we've seen
prediction_loss = vmap_criterion(probabilistic_preds, output).mean()
# Compute KL divergence term (its definition has been taken care of by BBVIConverter)
kl_loss = BBVI_model.kl_divergence()
# Compute the weight accounting for the mini batch size and number of variational parameters
kl_weight = input.shape[0]/BBVI_model.kl_denominator

# Sum losses and optimize
total_loss = prediction_loss + kl_loss*kl_weight
total_loss.backward()
optimizer.step()
```
For SVGD:
```python
### SVGD
# Compute mean loss as we've seen
prediction_loss = vmap_criterion(probabilistic_preds, output).mean()
# Populate original gradients
prediction_loss.backward()

# Perturb the original gradients multiplying them by the kernel matrix
SVGD_model.perturb_gradients()
# Compute kernel gradients
kernel_loss = model.kernel_matrix.sum(dim=1).mean()

# Accumulate gradients
kernel_loss.backward()
# Optimize the parameters
optimizer.step()
```

### Example
A test script (test.py) is provided to help you get started. It demostrates how to convert a simple feed-forward network to its Bayesian counterpart. The BNN is then used to approximate a toy deterministic function. You can try different the two different posterior approximation algorithms and override the other hyperparameters directly from the command line:
- `--algorithms`: choose between `svgd` and `bbvi`
- `--n_samples`: number of Monte Carlo realizations
- `--num_train`: number of train examples
- `--hidden_shape`, `--hidden_shape_2`: hydden layer sizes for the network (hidden_shape 16, hidden_shape_2)
- `learning_rate`, `batch_size`, `epochs`: training hyperparameters

## Advanced usage
- the `parameter_list` attribute of Converters() allows to perform Hybrid Bayesian conversion: users wanting to mix up deterministic networks with stochastic ones are free to specify layer names or parameter groups and the Converter will take of the rest: 
```python
# hybrid SVGD conversion
H_SVGD = SVGDConverter().convert(
    torch.nn.Linear(10, 5), parameter_list = ['weight'])
```
```python
# Hybrid BBVI conversion
H_BBVI = BBVIConverter().convert(
    SimpleModule(), parameter_list = ['fc1', 'fc3.bias'], GaussianParameter)
```
- for BBVI, users are free to define variational posterior as complex as they want, provided they adhere to the `forward()` sampling logic encapsulation, `kl_divergence()` method definition and include the original `torch.nn.Parameter` in the initialization.
- for SVGD, users can modify the initial distance of the particles by using the `particle_config` dictionary:
```python
SVGD_ = SVGDConverter().convert(
    SimpleModule(), GaussianParameter, particle_config = {'prior_std':0.01})
```

## Future plans
Right now, Wololo is only optimized for Dense architectures. This is due to its usage of `torch.vmap` to manage computations for multiple batches of parameters. This limitation will be the first we will address to allow for Bayesian Convolutional Network experimentation.
