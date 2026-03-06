"""Unit tests for the Trainer class."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, TensorDataset

from lmt.models.config import ModelConfigPresets
from lmt.models.gpt import GPT
from lmt.training.config import BaseTrainingConfig
from lmt.training.trainer import Trainer


class TestTrainer:
    """Test suite for the Trainer class."""

    def create_mock_data_loaders(
        self, batch_size=2, num_batches=3, seq_length=8, vocab_size=1000
    ):
        """Create mock data loaders for testing."""
        # Create dummy data
        total_samples = batch_size * num_batches
        input_data = torch.randint(0, vocab_size, (total_samples, seq_length))
        target_data = torch.randint(0, vocab_size, (total_samples, seq_length))

        # Create datasets and data loaders
        train_dataset = TensorDataset(input_data, target_data)
        val_dataset = TensorDataset(
            input_data[:batch_size], target_data[:batch_size]
        )

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=False
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )

        return train_loader, val_loader

    def test_trainer_initialization(self):
        """Test Trainer initializes correctly."""
        # Create model and config
        model_config = ModelConfigPresets.small_gpt(context_length=16)
        model = GPT(model_config)

        train_loader, val_loader = self.create_mock_data_loaders()

        training_config = BaseTrainingConfig(
            num_epochs=1,
            eval_freq=5,
            eval_iter=2,
            learning_rate=1e-4,
            weight_decay=0.01,
            device='cpu',
        )

        trainer = Trainer(model, train_loader, val_loader, training_config)

        # Check initialization
        assert trainer.model is model
        assert trainer.train_loader is train_loader
        assert trainer.val_loader is val_loader
        assert trainer.config is training_config
        assert trainer.device.type == 'cpu'
        assert isinstance(trainer.optimizer, torch.optim.AdamW)
        assert trainer.train_losses == []
        assert trainer.val_losses == []
        assert trainer.global_step == -1
        assert trainer.examples_seen == 0

    def test_trainer_model_device_placement(self):
        """Test that model is moved to correct device."""
        model_config = ModelConfigPresets.small_gpt(context_length=8)
        model = GPT(model_config)

        train_loader, val_loader = self.create_mock_data_loaders()

        training_config = BaseTrainingConfig(
            num_epochs=1,
            eval_freq=5,
            eval_iter=2,
            learning_rate=1e-4,
            weight_decay=0.01,
            device='cpu',
        )

        trainer = Trainer(model, train_loader, val_loader, training_config)

        # Check that model parameters are on the correct device
        for param in trainer.model.parameters():
            assert param.device.type == 'cpu'

    def test_trainer_train_step(self):
        """Test individual training step."""
        model_config = ModelConfigPresets.small_gpt(context_length=8)
        model = GPT(model_config)

        train_loader, val_loader = self.create_mock_data_loaders(
            seq_length=8, vocab_size=model_config.vocab_size
        )

        training_config = BaseTrainingConfig(
            num_epochs=1,
            eval_freq=5,
            eval_iter=2,
            learning_rate=1e-4,
            weight_decay=0.01,
            device='cpu',
        )

        trainer = Trainer(model, train_loader, val_loader, training_config)

        # Get a batch
        input_batch, target_batch = next(iter(train_loader))

        # Perform training step
        initial_examples_seen = trainer.examples_seen
        initial_global_step = trainer.global_step

        loss = trainer.train_step(input_batch, target_batch)

        # Check that loss is a tensor and counters are updated
        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0  # Loss should be positive
        assert trainer.examples_seen > initial_examples_seen
        assert trainer.global_step == initial_global_step + 1

    @patch('lmt.training.trainer.evaluate_model')
    def test_trainer_evaluate_step(self, mock_evaluate_model):
        """Test evaluation step."""
        mock_evaluate_model.return_value = (
            1.5,
            2.0,
        )  # Mock train_loss, val_loss

        model_config = ModelConfigPresets.small_gpt(context_length=8)
        model = GPT(model_config)

        train_loader, val_loader = self.create_mock_data_loaders()

        training_config = BaseTrainingConfig(
            num_epochs=1,
            eval_freq=5,
            eval_iter=2,
            learning_rate=1e-4,
            weight_decay=0.01,
            device='cpu',
        )

        trainer = Trainer(model, train_loader, val_loader, training_config)
        trainer.examples_seen = 100
        trainer.global_step = 10

        train_loss, val_loss = trainer.evaluate_step()

        assert train_loss == 1.5
        assert val_loss == 2.0
        assert len(trainer.track_examples_seen) == 1
        assert trainer.track_examples_seen[0] == 100
        assert len(trainer.track_global_steps) == 1
        assert trainer.track_global_steps[0] == 10

    def test_trainer_save_model(self):
        """Test model saving functionality."""
        model_config = ModelConfigPresets.small_gpt(context_length=8)
        model = GPT(model_config)

        train_loader, val_loader = self.create_mock_data_loaders()

        with tempfile.TemporaryDirectory() as temp_dir:
            training_config = BaseTrainingConfig(
                num_epochs=1,
                eval_freq=5,
                eval_iter=2,
                learning_rate=1e-4,
                weight_decay=0.01,
                device='cpu',
                save_dir=temp_dir,
            )

            trainer = Trainer(model, train_loader, val_loader, training_config)

            save_subdir = Path('test_weights')
            save_name = Path('test_model.pth')
            trainer.save_model(save_subdir, save_name)

            # Check that file was created
            save_path = Path(temp_dir) / save_subdir / save_name
            assert save_path.exists()

            # Check that saved file contains expected keys
            checkpoint = torch.load(save_path, map_location='cpu')
            assert 'model_state_dict' in checkpoint
            assert 'optimizer_state_dict' in checkpoint
            assert 'config' in checkpoint

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_trainer_plot_losses(self, mock_close, mock_savefig):
        """Test loss plotting functionality."""
        model_config = ModelConfigPresets.small_gpt(context_length=8)
        model = GPT(model_config)

        train_loader, val_loader = self.create_mock_data_loaders()

        with tempfile.TemporaryDirectory() as temp_dir:
            training_config = BaseTrainingConfig(
                num_epochs=1,
                eval_freq=5,
                eval_iter=2,
                learning_rate=1e-4,
                weight_decay=0.01,
                device='cpu',
                save_dir=temp_dir,
            )

            trainer = Trainer(model, train_loader, val_loader, training_config)

            # Add some mock loss data
            trainer.train_losses = [2.5, 2.0, 1.5]
            trainer.val_losses = [2.8, 2.3, 1.8]
            x_axis_data = [100, 200, 300]

            trainer.plot_losses(x_axis_data)

            # Check that matplotlib functions were called
            mock_savefig.assert_called_once()
            mock_close.assert_called_once()

    def test_trainer_plot_losses_no_data(self):
        """Test plot_losses with no data."""
        model_config = ModelConfigPresets.small_gpt(context_length=8)
        model = GPT(model_config)

        train_loader, val_loader = self.create_mock_data_loaders()

        training_config = BaseTrainingConfig(
            num_epochs=1,
            eval_freq=5,
            eval_iter=2,
            learning_rate=1e-4,
            weight_decay=0.01,
            device='cpu',
        )

        trainer = Trainer(model, train_loader, val_loader, training_config)

        # Should not raise an error with empty data
        trainer.plot_losses([])

    @patch('lmt.training.trainer.evaluate_model')
    def test_trainer_full_training_loop(self, mock_evaluate_model):
        """Test complete training loop."""
        mock_evaluate_model.return_value = (1.5, 2.0)

        model_config = ModelConfigPresets.small_gpt(context_length=8)
        model = GPT(model_config)

        train_loader, val_loader = self.create_mock_data_loaders(
            batch_size=1,
            num_batches=2,
            seq_length=8,
            vocab_size=model_config.vocab_size,
        )

        training_config = BaseTrainingConfig(
            num_epochs=1,
            eval_freq=1,  # Evaluate every step
            eval_iter=1,
            learning_rate=1e-4,
            weight_decay=0.01,
            device='cpu',
        )

        trainer = Trainer(model, train_loader, val_loader, training_config)

        with patch('builtins.print'):  # Suppress print statements
            result = trainer.train()

        # Check return values
        assert 'train_losses' in result
        assert 'val_losses' in result
        assert 'execution_time' in result
        assert 'track_examples_seen' in result
        assert 'track_global_steps' in result

        # Check that training actually happened
        assert len(result['train_losses']) > 0
        assert len(result['val_losses']) > 0
        assert result['execution_time'] > 0
        assert trainer.global_step > -1

    def test_trainer_with_tokenizer(self):
        """Test Trainer initialization with tokenizer."""
        from lmt.tokenizer.naive import NaiveTokenizer

        model_config = ModelConfigPresets.small_gpt(context_length=8)
        model = GPT(model_config)

        train_loader, val_loader = self.create_mock_data_loaders()

        vocab = {'hello': 0, 'world': 1, '<unk>': 2}
        tokenizer = NaiveTokenizer(vocab)

        training_config = BaseTrainingConfig(
            num_epochs=1,
            eval_freq=5,
            eval_iter=2,
            learning_rate=1e-4,
            weight_decay=0.01,
            device='cpu',
        )

        trainer = Trainer(
            model, train_loader, val_loader, training_config, tokenizer
        )

        assert trainer.tokenizer is tokenizer

    def test_trainer_optimizer_parameters(self):
        """Test that optimizer is configured correctly."""
        model_config = ModelConfigPresets.small_gpt(context_length=8)
        model = GPT(model_config)

        train_loader, val_loader = self.create_mock_data_loaders()

        lr = 1e-3
        weight_decay = 0.05

        training_config = BaseTrainingConfig(
            num_epochs=1,
            eval_freq=5,
            eval_iter=2,
            learning_rate=lr,
            weight_decay=weight_decay,
            device='cpu',
        )

        trainer = Trainer(model, train_loader, val_loader, training_config)

        # Check optimizer configuration
        assert isinstance(trainer.optimizer, torch.optim.AdamW)

        # Check optimizer parameters
        optimizer_params = trainer.optimizer.param_groups[0]
        assert optimizer_params['lr'] == lr
        assert optimizer_params['weight_decay'] == weight_decay

    def test_trainer_global_step_increment(self):
        """Test that global step increments exactly once per batch."""
        model_config = ModelConfigPresets.small_gpt(context_length=8)
        model = GPT(model_config)

        train_loader, val_loader = self.create_mock_data_loaders(
            batch_size=1,
            num_batches=3,
            seq_length=8,
            vocab_size=model_config.vocab_size,
        )

        training_config = BaseTrainingConfig(
            num_epochs=1,
            eval_freq=10,  # Don't evaluate during this test
            eval_iter=1,
            learning_rate=1e-4,
            weight_decay=0.01,
            device='cpu',
        )

        trainer = Trainer(model, train_loader, val_loader, training_config)

        # global_step starts at -1, increments to 0 on first step
        assert trainer.global_step == -1

        for i, (input_batch, target_batch) in enumerate(train_loader):
            trainer.optimizer.zero_grad()
            loss = trainer.train_step(input_batch, target_batch)
            loss.backward()
            trainer.optimizer.step()

            # train_step increments global_step by exactly 1
            assert trainer.global_step == i, (
                f'After {i + 1} steps, global_step should be {i} '
                f'but got {trainer.global_step}'
            )
