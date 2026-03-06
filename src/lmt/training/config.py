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
"""Module for training configuration.

This module defines the BaseTrainingConfig class, which is used to manage
configuration parameters for a machine learning training pipeline.
"""


class BaseTrainingConfig:
    """Base configuration class for defining training parameters.

    This class provides a standard set of parameters required for
    configuring a training run, such as the number of epochs,
    evaluation frequency, learning rate, and device. It also supports
    additional custom parameters via keyword arguments.

    Attributes:
        num_epochs (int): The total number of epochs to train for.
        eval_freq (int): The frequency (in steps) at which to run evaluation.
        eval_iter (int): The number of evaluation steps to run.
        learning_rate (float): The learning rate for the optimizer.
        weight_decay (float): The weight decay for the optimizer.
        batch_size (int): The number of samples per batch.
        device (str): The device to use for training
            (e.g., 'cpu', 'cuda', 'mps').
        save_dir (str): The directory where training artifacts will be saved.
    """

    def __init__(
        self,
        num_epochs: int,
        eval_freq: int,
        eval_iter: int,
        learning_rate: float,
        weight_decay: float,
        batch_size: int = 2,
        device: str = 'mps',
        save_dir: str = 'runs',
        task: str = 'pretraining',
        **kwargs,
    ):
        """Initializes the BaseTrainingConfig with training parameters.

        Args:
            num_epochs (int): The total number of epochs to train for.
            eval_freq (int): The frequency (in steps) at which to run
                evaluation.
            eval_iter (int): The number of evaluation steps to run.
            learning_rate (float): The learning rate for the optimizer.
            weight_decay (float): The weight decay for the optimizer.
            batch_size (int, optional): The number of samples per batch.
                Defaults to 2.
            device (str, optional): The device to use for training.
                Defaults to 'mps'.
            save_dir (str, optional): The directory where training artifacts
                will be saved. Defaults to 'runs'.
            task (str): Training task (e.g. pretraining or classification)
            **kwargs (Any): Additional keyword arguments stored as attributes.
        """
        self.num_epochs = num_epochs
        self.eval_freq = eval_freq
        self.eval_iter = eval_iter
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.device = device
        self.save_dir = save_dir
        self.task = task

        # Store any additional config parameters
        for key, value in kwargs.items():
            setattr(self, key, value)
