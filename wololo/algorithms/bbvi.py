import torch


def bbvi_initialize(module, stochastic_parameter) -> None:
    """
    Initializes the BBVI-specific attributes for the module:
    - Prepares a list of stochastic parameters.
    - Computes the KL divergence normalization denominator.
    """
    # Prepare list of stochastic parameters
    module.stochastic_parameters = [
        (name, module)
        for name, module in module.named_modules()
        if isinstance(module, stochastic_parameter)
    ]

    # Compute and add KL denominator
    kl_denominator = sum(
        p.numel() for _, sp in module.stochastic_parameters for p in sp.parameters()
    )
    module.kl_denominator = kl_denominator


def bbvi_kl_divergence(module) -> torch.Tensor:
    """
    Computes the total KL divergence for all stochastic parameters in the module.
    """
    kl_div = 0
    for name, stochastic_param in module.stochastic_parameters:
        kl_div += stochastic_param.kl_divergence()
    return kl_div
