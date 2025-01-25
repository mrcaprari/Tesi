import torch
from tqdm import tqdm

from ..converters import Converter


def BBVI(
    BBVI_model,
    n_samples,
    epochs,
    train_loader,
    loss_fn,
    optimizer_fn,
    learning_rate,
    validation_loader=None,
    test_loader=None,
    task="regression",  # "regression" or "classification"
    scheduler_fn=None,
    grad_clip=None,
):
    """
    Perform Black Box Variational Inference (BBVI) on the given model.

    Args:
        BBVI_model: The stochastic model to be optimized.
        n_samples (int): Number of random parameter realizations to use during inference.
        epochs (int): Number of training epochs.
        train_loader: PyTorch DataLoader for training data.
        loss_fn: Loss function used to compute prediction loss.
        optimizer_fn: Function to instantiate the optimizer.
        learning_rate (float): Learning rate for the optimizer.
        validation_loader: PyTorch DataLoader for validation data. Defaults to None.
        test_loader: PyTorch DataLoader for test data. Defaults to None.
        task (str): Task type, either "regression" or "classification". Defaults to "regression".
        scheduler_fn: Function to instantiate the learning rate scheduler. Defaults to None.
        grad_clip (float, optional): Gradient clipping value. Defaults to None.

    Returns:
        Tuple:
            - train_history (dict): History of training losses and metrics.
            - val_history (dict): History of validation losses and metrics.
            - test_history (dict): History of test losses and metrics.
    """
    optimizer = optimizer_fn(BBVI_model.parameters(), lr=learning_rate)
    scheduler = scheduler_fn(optimizer) if scheduler_fn is not None else None

    train_history = {"pred_loss": [], "kl_loss": [], "total_loss": []}
    val_history = {"pred_loss": [], "kl_loss": [], "total_loss": []}
    test_history = {"pred_loss": [], "kl_loss": [], "total_loss": []}

    if task == "classification":
        train_history["accuracy"] = []
        val_history["accuracy"] = []
        test_history["accuracy"] = []

    num_parameters = sum(p.numel() for p in BBVI_model.parameters() if p.requires_grad)

    for epoch in range(epochs):
        # Training
        BBVI_model.train()
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{epochs}") as pbar:
            for x_batch, y_batch in train_loader:
                batch_size = x_batch.shape[0]
                kl_weight = batch_size / num_parameters

                pred_loss, kl_loss, total_loss = BBVI_step(
                    BBVI_model,
                    n_samples,
                    x_batch,
                    y_batch,
                    loss_fn,
                    optimizer,
                    kl_weight,
                    grad_clip,
                )

                train_history["pred_loss"].append(pred_loss.detach().cpu().numpy())
                train_history["kl_loss"].append(kl_loss.detach().cpu().numpy())
                train_history["total_loss"].append(total_loss.detach().cpu().numpy())

                if task == "classification":
                    accuracy = BBVI_accuracy(BBVI_model, x_batch, y_batch, n_samples)
                    train_history["accuracy"].append(accuracy)

                pbar.set_postfix(
                    tot_loss=total_loss.item(),
                    pred=pred_loss.item(),
                    kl=kl_loss.item(),
                )
                pbar.update(1)

        # Validation
        if validation_loader is not None:
            BBVI_model.eval()
            val_pred_loss, val_kl_loss, val_accuracy = BBVI_evaluate(
                BBVI_model, n_samples, validation_loader, loss_fn, task, num_parameters
            )
            val_total_loss = val_pred_loss + val_kl_loss

            val_history["pred_loss"].append(val_pred_loss.detach().cpu().numpy())
            val_history["kl_loss"].append(val_kl_loss.detach().cpu().numpy())
            val_history["total_loss"].append(val_total_loss.detach().cpu().numpy())

            if task == "classification":
                val_history["accuracy"].append(val_accuracy)

        # Update learning rate
        if scheduler is not None:
            scheduler.step()

    # Testing
    if test_loader is not None:
        BBVI_model.eval()
        test_pred_loss, test_kl_loss, test_accuracy = BBVI_evaluate(
            BBVI_model, n_samples, test_loader, loss_fn, task, num_parameters
        )

        test_total_loss = test_pred_loss + test_kl_loss

        test_history["pred_loss"].append(test_pred_loss.detach().cpu().numpy())
        test_history["kl_loss"].append(test_kl_loss.detach().cpu().numpy())
        test_history["total_loss"].append(test_total_loss.detach().cpu().numpy())

        if task == "classification":
            test_history["accuracy"].append(test_accuracy)

    return train_history, val_history, test_history


def BBVI_step(model, n_samples, x, y, loss_fn, optimizer, kl_weight, grad_clip=None):
    """
    Perform a single step of Black Box Variational Inference (BBVI).

    Args:
        model: The stochastic model being trained.
        n_samples (int): Number of random parameter realizations to use during inference.
        x (torch.Tensor): Input batch of data.
        y (torch.Tensor): Target batch of data.
        loss_fn: Loss function used to compute prediction loss.
        optimizer: Optimizer instance for updating model parameters.
        kl_weight (float): Weight applied to the KL divergence loss.
        grad_clip (float, optional): Gradient clipping value. Defaults to None.

    Returns:
        Tuple:
            - pred_loss (torch.Tensor): Prediction loss for the batch.
            - kl_loss (torch.Tensor): KL divergence loss for the batch.
            - total_loss (torch.Tensor): Total loss (prediction + KL divergence) for the batch.
    """
    optimizer.zero_grad()
    output = model(x, n_samples)
    pred_loss = torch.vmap(loss_fn, in_dims=(0, None))(output, y).mean()

    kl_loss = model.kl_divergence() * kl_weight
    total_loss = pred_loss + kl_loss

    total_loss.backward()

    # Gradient clipping
    if grad_clip is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    optimizer.step()

    return pred_loss, kl_loss, total_loss


def BBVI_evaluate(model, n_samples, dataloader, loss_fn, task, num_parameters):
    """
    Evaluate the model on a given dataset.

    Args:
        model: The stochastic model being evaluated.
        n_samples (int): Number of random parameter realizations to use during inference.
        dataloader: PyTorch DataLoader for evaluation data.
        loss_fn: Loss function used to compute prediction loss.
        task (str): Task type, either "regression" or "classification".
        num_parameters (int): Total number of trainable parameters in the model.

    Returns:
        Tuple:
            - pred_loss (torch.Tensor): Prediction loss for the dataset.
            - kl_loss (torch.Tensor): KL divergence loss for the dataset.
            - accuracy (float, optional): Accuracy for classification tasks.
    """
    pred_losses = []
    kl_losses = []
    accuracies = []

    for x_batch, y_batch in dataloader:
        batch_size = x_batch.shape[0]
        kl_weight = batch_size / num_parameters

        output = model(x_batch, n_samples)
        pred_loss = torch.vmap(loss_fn, in_dims=(0, None))(output, y_batch).mean()
        kl_loss = model.kl_divergence() * kl_weight

        pred_losses.append(pred_loss.detach().cpu().numpy())
        kl_losses.append(kl_loss.detach().cpu().numpy())

        if task == "classification":
            accuracy = BBVI_accuracy(model, x_batch, y_batch, n_samples)
            accuracies.append(accuracy)

    pred_loss = torch.tensor(pred_losses).mean()
    kl_loss = torch.tensor(kl_losses).mean()
    accuracy = torch.tensor(accuracies).mean() if task == "classification" else None

    return pred_loss, kl_loss, accuracy


def BBVI_accuracy(model, x, y, n_samples):
    """
    Compute accuracy for classification tasks.

    Args:
        model: The stochastic model being evaluated.
        x (torch.Tensor): Input batch of data.
        y (torch.Tensor): Target batch of data.
        n_samples (int): Number of random parameter realizations to use during inference.

    Returns:
        float: Accuracy for the batch.
    """
    output = model(x, n_samples)
    predictions = output.argmax(dim=-1)
    accuracy = (predictions == y).float().mean()
    return accuracy


# def BBVI(
#     starting_model,
#     stochastic_parameter,
#     n_samples,
#     epochs,
#     dataloader,
#     loss_fn,
#     optimizer_fn,
#     learning_rate,
#     transform_list=[],
# ):
#     """
#     Perform Black Box Variational Inference (BBVI) on the given model.

#     Args:
#         starting_model: The initial model to be optimized.
#         stochastic_parameter: A module encoding the random parameter logic,
#             which will substitute parameters specified in the `transform_list` with corresponding
#             stochastic modules.
#         transform_list (list): List of parameters apply the random transformation to.
#         n_samples (int): Number of random parameter realizations to use during inference.
#         epochs (int): Number of training epochs.
#         dataloader: PyTorch DataLoader providing training data batches.
#         loss_fn: Loss function used to compute prediction loss.
#         optimizer_fn: Function to instantiate the optimizer.
#         learning_rate (float): Learning rate for the optimizer.

#     Returns:
#         Tuple:
#             - model: The trained model after BBVI.
#             - pred_history (list): History of prediction losses for all batches.
#             - kl_history (list): History of KL divergence losses for all batches.
#             - total_history (list): History of total losses for all batches.
#     """
#     num_parameters = sum(
#         p.numel() for p in starting_model.parameters() if p.requires_grad
#     )
#     model = Converter().convert(starting_model, stochastic_parameter, transform_list)

#     optimizer = optimizer_fn(model.parameters(), lr=learning_rate)

#     pred_history = []
#     kl_history = []
#     total_history = []

#     for epoch in range(epochs):
#         with tqdm(total=len(dataloader), desc=f"Epoch {epoch + 1}/{epochs}") as pbar:
#             for x_batch, y_batch in dataloader:
#                 batch_size = x_batch.shape[0]
#                 kl_weight = batch_size / num_parameters

#                 pred_loss, kl_loss, total_loss = BBVI_step(
#                     model,
#                     n_samples,
#                     x_batch,
#                     y_batch,
#                     loss_fn,
#                     optimizer,
#                     kl_weight,
#                 )

#                 pred_history.append(pred_loss.detach().cpu().numpy())
#                 kl_history.append(kl_loss.detach().cpu().numpy())
#                 total_history.append(total_loss.detach().cpu().numpy())

#                 pbar.set_postfix(
#                     tot_loss=total_loss.item(),
#                     pred=pred_loss.item(),
#                     kernel=kl_loss.item(),
#                 )
#                 pbar.update(1)

#     return model, pred_history, kl_history, total_history


# def BBVI_step(model, n_samples, x, y, loss_fn, optimizer, kl_weight):
#     """
#     Perform a single step of Black Box Variational Inference (BBVI).

#     Args:
#         model: The stochastic model being trained.
#         n_samples (int): Number of random parameter realizations to use during inference.
#         x (torch.Tensor): Input batch of data.
#         y (torch.Tensor): Target batch of data.
#         loss_fn: Loss function used to compute prediction loss.
#         optimizer: Optimizer instance for updating model parameters.
#         kl_weight (float): Weight applied to the KL divergence loss.

#     Returns:
#         Tuple:
#             - pred_loss (torch.Tensor): Prediction loss for the batch.
#             - kl_loss (torch.Tensor): KL divergence loss for the batch.
#             - total_loss (torch.Tensor): Total loss (prediction + KL divergence) for the batch.
#     """
#     optimizer.zero_grad()
#     output = model(x, n_samples)
#     pred_loss = torch.vmap(loss_fn, in_dims=(0, None))(output, y).mean()

#     kl_loss = model.kl_divergence()
#     kl_loss = kl_loss * kl_weight

#     total_loss = pred_loss + kl_loss
#     total_loss.backward()
#     optimizer.step()
#     return pred_loss, kl_loss, total_loss
