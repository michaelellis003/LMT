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
"""Trainer classes for training logic.

This module contains the core classes for managing the training
and evaluation loops of a machine learning model.
"""

import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from lmt.tokenizer.base import BaseTokenizer
from lmt.training.config import BaseTrainingConfig
from lmt.training.curriculum import CurriculumSchedule
from lmt.training.loss import calc_loss_batch, evaluate_model


class Trainer:
    """Trainer class for pretraining and finetuning.

    This class provides a foundation for training a PyTorch model. It
    handles model and data loader initialization, device placement,
    optimizer setup, and basic state tracking for training.

    Attributes:
        model: The PyTorch model to train, moved to the specified device.
        train_loader: The data loader for the training set.
        val_loader: The data loader for the validation set.
        config: The configuration object for training.
        tokenizer: The tokenizer instance, if provided.
        device: The device (e.g., 'cuda', 'cpu') the model is on.
        optimizer: The optimizer for updating model weights.
        train_losses: A list to store training loss values per step.
        val_losses: A list to store validation loss values.
        global_step: A counter for the number of training steps completed.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: BaseTrainingConfig,
        tokenizer: BaseTokenizer | None = None,
        curriculum: CurriculumSchedule | None = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    ):
        """Initializes the Trainer.

        Args:
            model: The PyTorch model (nn.Module) to be trained.
            train_loader: A DataLoader for the training dataset.
            val_loader: A DataLoader for the validation dataset.
            config: A BaseTrainingConfig object containing training
                hyperparameters.
            tokenizer: An optional BaseTokenizer instance for text processing.
            curriculum: An optional curriculum schedule that controls
                sequence length during training. When provided, batches
                are truncated according to the schedule.
            scheduler: An optional PyTorch LR scheduler. When provided,
                ``scheduler.step()`` is called after each optimizer step.
                Use any ``torch.optim.lr_scheduler`` class.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.tokenizer = tokenizer
        self.curriculum = curriculum
        self.scheduler = scheduler

        self.device = torch.device(config.device)
        self.model.to(self.device)

        # Initialize optimizer
        # TODO: Generalize to other optimizers
        #       consider adding optimizer_type to config
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Training tracking
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []

        self.examples_seen = 0
        self.track_examples_seen: list[int] = []

        self.global_step = -1
        self.track_global_steps: list[int] = []

        # MLflow logging (enabled when run_name is set)
        self._mlflow_active = False
        run_name = getattr(config, 'run_name', None)
        if run_name:
            import mlflow

            self._mlflow = mlflow
            tracking_uri = Path(config.save_dir).resolve() / 'mlruns'
            mlflow.set_tracking_uri(f'file://{tracking_uri}')
            mlflow.set_experiment('lmt')
            mlflow.start_run(run_name=run_name)
            self._mlflow_active = True

            # Log training config as params
            param_count = sum(p.numel() for p in model.parameters())
            mlflow.log_params(
                {
                    'num_epochs': config.num_epochs,
                    'learning_rate': config.learning_rate,
                    'weight_decay': config.weight_decay,
                    'batch_size': config.batch_size,
                    'eval_freq': config.eval_freq,
                    'max_grad_norm': str(config.max_grad_norm),
                    'task': config.task,
                    'device': config.device,
                    'param_count': param_count,
                }
            )

    def train_step(
        self, input_batch: torch.Tensor, target_batch: torch.Tensor
    ) -> torch.Tensor:
        """Performs a single training step.

        Calculates the loss for a batch, updates the `examples_seen` and
        `global_step` counters.

        Args:
            input_batch: A `torch.Tensor` containing the input data for the
                batch.
            target_batch: A `torch.Tensor` containing the target data for the
                batch.

        Returns:
            A `torch.Tensor` representing the calculated loss for the batch.
        """
        loss = calc_loss_batch(
            input_batch,
            target_batch,
            self.model,
            self.device,
            self.config.task,
            aux_loss_coeff=self.config.aux_loss_coeff,
        )
        self.examples_seen += input_batch.shape[0]
        self.global_step += 1
        return loss

    def evaluate_step(self) -> tuple[float, float]:
        """Performs an evaluation step for pretraining.

        Evaluates the model on both the training and validation datasets,
        then records the current number of tokens seen and global steps.

        Returns:
            A tuple containing:
                - The average training loss (`float`).
                - The average validation loss (`float`).
        """
        train_loss, val_loss = evaluate_model(
            self.model,
            self.train_loader,
            self.val_loader,
            self.device,
            self.config.eval_iter,
            self.config.task,
        )
        self.track_examples_seen.append(self.examples_seen)
        self.track_global_steps.append(self.global_step)
        return train_loss, val_loss

    def train(self) -> dict[str, Any]:
        """Runs the main training loop for the model.

        This method iterates through the specified number of epochs,
        performing training steps and periodic evaluations. It tracks
        training and validation losses and prints progress.

        Returns:
            A dictionary containing the final training losses,
            validation losses, and the total execution time in minutes.
        """
        print('Starting model training...')
        start_time = time.time()

        try:
            return self._run_training_loop(start_time)
        except BaseException:
            if self._mlflow_active:
                self._mlflow.end_run(status='FAILED')
                self._mlflow_active = False
            raise

    def _run_training_loop(self, start_time: float) -> dict[str, Any]:
        """Execute the training loop.

        Args:
            start_time: Training start time from ``time.time()``.

        Returns:
            Dictionary with losses and execution metadata.
        """
        for epoch in range(self.config.num_epochs):
            self.model.train()

            for input_batch, target_batch in self.train_loader:
                # Apply curriculum: truncate sequences early in training
                if self.curriculum is not None:
                    input_batch, target_batch = self.curriculum.truncate_batch(
                        input_batch,
                        target_batch,
                        step=self.global_step + 1,
                    )

                self.optimizer.zero_grad()
                loss = self.train_step(input_batch, target_batch)
                loss.backward()
                if self.config.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm,
                    )
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()

                # Log step-level metrics to MLflow
                if self._mlflow_active:
                    step_metrics: dict[str, float] = {
                        'train_step_loss': loss.item(),
                    }
                    if self.curriculum is not None:
                        step_metrics['curriculum_seq_length'] = (
                            self.curriculum.get_length(self.global_step)
                        )
                    if self.scheduler is not None:
                        step_metrics['lr'] = self.optimizer.param_groups[0][
                            'lr'
                        ]
                    self._mlflow.log_metrics(
                        step_metrics, step=self.global_step
                    )

                # Periodic evaluation
                if self.global_step % self.config.eval_freq == 0:
                    train_loss, val_loss = self.evaluate_step()
                    self.train_losses.append(train_loss)
                    self.val_losses.append(val_loss)

                    # Log eval losses to MLflow
                    if self._mlflow_active:
                        self._mlflow.log_metrics(
                            {
                                'train_loss': train_loss,
                                'val_loss': val_loss,
                            },
                            step=self.global_step,
                        )

                    print(
                        f'Ep {epoch + 1} (Step {self.global_step:06d}): '
                        f'Train loss: {train_loss:.3f}, '
                        f'Val loss: {val_loss:.3f}'
                    )

        end_time = time.time()
        execution_time = (end_time - start_time) / 60

        # End MLflow run
        if self._mlflow_active:
            self._mlflow.end_run()
            self._mlflow_active = False

        print(f'Training completed in {execution_time:.2f} minutes.')

        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'execution_time': execution_time,
            'track_examples_seen': self.track_examples_seen,
            'track_global_steps': self.track_global_steps,
        }

    def save_model(
        self,
        save_subdir: Path = Path('model_weights'),
        save_name: Path = Path('model_and_optimizer.pth'),
    ) -> None:
        """Save model and optimizer state.

        Args:
            save_subdir (Path):  The directory where the weights will be saved.
            save_name (Path): File name
        """
        save_path = Path(self.config.save_dir).joinpath(save_subdir, save_name)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        print(f'Saving model and optimizer to {save_path}')
        torch.save(
            {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'config': self.config.__dict__,
            },
            save_path,
        )

    def plot_losses(
        self,
        x_axis_data: list[int],
        save_subdir: Path = Path('plots'),
        save_name: Path = Path('loss_plot.png'),
    ) -> None:
        """Plots training and validation losses and saves the figure.

        Args:
            x_axis_data: Data for x-axis (e.g. number of training steps).
            save_subdir (Path): The directory where the plot will be saved.
            save_name (Path): File name
        """
        if not x_axis_data:
            print('No data to plot. Skipping loss plot generation.')
            return

        import matplotlib.pyplot as plt

        # Create a new figure and axes for the plot
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot training and validation losses
        ax.plot(x_axis_data, self.train_losses, label='Training Loss')
        ax.plot(x_axis_data, self.val_losses, label='Validation Loss')

        # Set plot title and labels
        ax.set_title('Training & Validation Loss Over Time')
        ax.set_xlabel('Examples Seen')
        ax.set_ylabel('Loss')

        # Add a legend and grid
        ax.legend()
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Save the plot to the specified directory
        save_path = Path(self.config.save_dir).joinpath(save_subdir, save_name)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        print(f'Saving loss plot to {save_path}')
        plt.savefig(save_path)
        plt.close(fig)  # Close the figure to free up memory
