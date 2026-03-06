# Copyright 2025 Michael Ellis
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Functions for calculating and evaluating model loss.

This module contains functions for calculating the loss on a single batch
of data, computing the average loss over a DataLoader, and evaluating a model's
performance on subsets of training and validation data.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def calc_loss_batch(
    input_batch_ids: torch.Tensor,
    target_batch_ids: torch.Tensor,
    model: nn.Module,
    device: torch.device,
    task_type: str = 'pretraining',
    aux_loss_coeff: float = 0.0,
) -> torch.Tensor:
    """Calculates loss for a batch based on task type.

    Optionally adds auxiliary loss from MoE models. If the model has
    an ``aux_loss`` attribute (e.g. Mixtral, DeepSeek-V2), it is
    scaled by ``aux_loss_coeff`` and added to the main loss.

    Args:
        input_batch_ids (torch.Tensor): The input tensor containing token IDs
            for the model. Shape: (batch_size, sequence_length).
        target_batch_ids (torch.Tensor): The target tensor containing the
            ground truth token IDs. Shape: (batch_size, sequence_length).
        model (torch.nn.Module): The neural network model that will produce
            the logits.
        device (torch.device): The device on which to perform the calculations.
        task_type (str): Either 'pretraining' or 'classification'
        aux_loss_coeff (float): Coefficient for auxiliary loss from MoE
            models. Set to 0.0 to disable. Typical value: 0.01.

    Returns:
        torch.Tensor: A scalar tensor representing the calculated loss.
    """
    input_batch = input_batch_ids.to(device)
    target_batch = target_batch_ids.to(device)

    match task_type:
        case 'pretraining':
            # For pretraining: predict next tokens
            logits = model(input_batch)
            loss = nn.functional.cross_entropy(
                logits.flatten(0, 1), target_batch.flatten()
            )
        case 'classification':
            # For classification: use last token's logits
            logits = model(input_batch)[:, -1, :]
            loss = nn.functional.cross_entropy(logits, target_batch)
        case _:
            raise ValueError(f'Unknown task_type: {task_type}')

    # Add auxiliary loss from MoE models if available
    if aux_loss_coeff > 0.0 and hasattr(model, 'aux_loss'):
        aux_loss: torch.Tensor = model.aux_loss  # type: ignore[assignment]
        loss = loss + aux_loss_coeff * aux_loss

    return loss


def calc_loss_loader(
    data_loader: DataLoader,
    model: nn.Module,
    device: torch.device,
    num_batches: int | None = None,
    task_type: str = 'pretraining',
) -> float:
    """Calculates the average loss over a specified number of batches.

    This function iterates through a DataLoader for a given number of batches,
    computes the loss for each batch, and returns the average loss. It's
    useful for calculating validation or test loss without necessarily
    running through the entire dataset.

    Args:
        data_loader (DataLoader): The PyTorch DataLoader to iterate over.
        model (nn.Module): The model to be evaluated.
        device (torch.device): The device (e.g., 'cuda' or 'cpu') on which to
            perform the calculations.
        num_batches (Optional[int], optional): The number of batches to use for
            calculating the loss. If None, all batches in the data_loader
            are used. Defaults to None.
        task_type (str): Either 'pretraining' or 'classification'

    Returns:
        float: The average loss per batch. Returns float("nan") if the
            data_loader is empty.
    """
    total_loss = 0.0
    if len(data_loader) == 0:
        return float('nan')

    num_batches = min(num_batches or len(data_loader), len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break
        loss = calc_loss_batch(
            input_batch, target_batch, model, device, task_type
        )

        total_loss += loss.item()

    return total_loss / num_batches


def evaluate_model(
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    device: torch.device,
    eval_iter: int,
    task_type: str = 'pretraining',
) -> tuple[float, float]:
    """Evaluates the model on training and validation data subsets.

    This function sets the model to evaluation mode (`model.eval()`),
    calculates the loss on a specified number of batches from both the training
    and validation dataloaders, and then returns the model to training mode
    (`model.train()`). This is typically used for periodic evaluation during a
    training loop.

    Args:
        model (nn.Module): The model to be evaluated.
        train_dataloader (DataLoader): The DataLoader for the training set.
        val_dataloader (DataLoader): The DataLoader for the validation set.
        device (torch.device): The device on which to perform calculations.
        eval_iter (int): The number of batches to use for the evaluation from
            each dataloader.
        task_type (str): Either 'pretraining' or 'classification'

    Returns:
        Tuple[float, float]: A tuple containing the average training loss and
            the average validation loss for the evaluated batches.
    """
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_dataloader,
            model,
            device,
            num_batches=eval_iter,
            task_type=task_type,
        )
        val_loss = calc_loss_loader(
            val_dataloader,
            model,
            device,
            num_batches=eval_iter,
            task_type=task_type,
        )
    model.train()
    return train_loss, val_loss
