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
"""Functions for saving and loading PyTorch model checkpoints.

This module provides utilities to save and load comprehensive model
checkpoints, including the model's state, optimizer's state, and configuration.
"""

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

# ============================================================================
# MODEL SAVING AND LOADING
# ============================================================================


def save_model_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: Any,
    save_path: Path,
    epoch: int | None = None,
    additional_info: dict[str, Any] | None = None,
):
    """Save a comprehensive model checkpoint.

    This function creates a dictionary containing the model's state,
    optimizer's state, and the configuration object. It then saves this
    dictionary to the specified file path using `torch.save`. The parent
    directories for the `save_path` will be created if they do not exist.

    Args:
        model (nn.Module): The PyTorch model to save.
        optimizer (torch.optim.Optimizer): The optimizer associated with the
            model.
        config (Any): The configuration object or dictionary for the model.
            If the object has a `__dict__` attribute, it will be used.
        save_path (Path): The Path object where the checkpoint will be saved.
        epoch (int, optional): The current epoch number to be saved.
            Defaults to None.
        additional_info (dict, optional): A dictionary of extra information to
            be included in the checkpoint. Defaults to None.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config.__dict__ if hasattr(config, '__dict__') else config,
    }

    if epoch is not None:
        checkpoint['epoch'] = epoch

    if additional_info:
        checkpoint.update(additional_info)

    torch.save(checkpoint, save_path)
    print(f'Model checkpoint saved to {save_path}')


def load_model_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Load a model checkpoint.

    This function loads a checkpoint dictionary from a file and updates the
    provided model and optimizer with the saved state. The checkpoint is loaded
    onto the specified device.

    Args:
        model (nn.Module): The PyTorch model to load the state into.
        checkpoint_path (Path): The Path object of the checkpoint file.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load the
            state into. Defaults to None.
        device (torch.device, optional): The device to map the checkpoint to.
            Defaults to 'cpu'.

    Returns:
        dict[str, Any]: The complete checkpoint dictionary.
    """
    if device is None:
        device = torch.device('cpu')

    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=True
    )

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print(f'Model checkpoint loaded from {checkpoint_path}')

    return checkpoint
