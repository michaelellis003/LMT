"""Integration tests for end-to-end training workflows."""

import tempfile
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from lmt.models.config import ModelConfig, ModelConfigPresets
from lmt.models.gpt import GPT
from lmt.tokenizer.bpe import BPETokenizer
from lmt.training.config import BaseTrainingConfig
from lmt.training.trainer import Trainer


class TestEndToEndTraining:
    """Integration tests for complete training workflows."""

    def create_synthetic_dataset(
        self, vocab_size=1000, seq_length=16, num_samples=50
    ):
        """Create a synthetic dataset for testing."""
        inputs = torch.randint(0, vocab_size, (num_samples, seq_length))
        targets = torch.randint(0, vocab_size, (num_samples, seq_length))
        return TensorDataset(inputs, targets)

    def test_small_model_training_workflow(self):
        """Test complete training workflow with a small model."""
        # Configure small model for fast testing
        model_config = ModelConfigPresets.small_gpt(context_length=16)
        model = GPT(model_config)

        # Create synthetic datasets
        train_dataset = self.create_synthetic_dataset(
            vocab_size=model_config.vocab_size, seq_length=8, num_samples=20
        )
        val_dataset = self.create_synthetic_dataset(
            vocab_size=model_config.vocab_size, seq_length=8, num_samples=10
        )

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

        # Configure training
        with tempfile.TemporaryDirectory() as temp_dir:
            training_config = BaseTrainingConfig(
                num_epochs=2,
                eval_freq=3,
                eval_iter=2,
                learning_rate=1e-3,
                weight_decay=0.01,
                device='cpu',
                save_dir=temp_dir,
            )

            # Create trainer and run training
            trainer = Trainer(model, train_loader, val_loader, training_config)

            # Store initial loss for comparison
            initial_params = {
                name: param.clone() for name, param in model.named_parameters()
            }

            # Run training
            result = trainer.train()

            # Verify training completed successfully
            assert 'train_losses' in result
            assert 'val_losses' in result
            assert 'execution_time' in result
            assert len(result['train_losses']) > 0
            assert len(result['val_losses']) > 0
            assert result['execution_time'] > 0

            # Verify model parameters changed (learning occurred)
            params_changed = False
            for name, param in model.named_parameters():
                if not torch.allclose(initial_params[name], param, atol=1e-6):
                    params_changed = True
                    break
            assert params_changed, (
                'Model parameters should change during training'
            )

            # Test model saving
            trainer.save_model()
            save_path = (
                Path(temp_dir) / 'model_weights' / 'model_and_optimizer.pth'
            )
            assert save_path.exists()

            # Test loss plotting
            trainer.plot_losses(trainer.track_examples_seen)

    def test_model_inference_after_training(self):
        """Test that model can be used for inference after training."""
        model_config = ModelConfigPresets.small_gpt(context_length=16)
        model = GPT(model_config)

        # Quick training
        train_dataset = self.create_synthetic_dataset(
            vocab_size=model_config.vocab_size, seq_length=8, num_samples=10
        )
        val_dataset = self.create_synthetic_dataset(
            vocab_size=model_config.vocab_size, seq_length=8, num_samples=5
        )

        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

        training_config = BaseTrainingConfig(
            num_epochs=1,
            eval_freq=5,
            eval_iter=1,
            learning_rate=1e-3,
            weight_decay=0.01,
            device='cpu',
        )

        trainer = Trainer(model, train_loader, val_loader, training_config)
        trainer.train()

        # Test inference
        model.eval()
        test_input = torch.randint(0, model_config.vocab_size, (1, 8))

        with torch.no_grad():
            output = model(test_input)

        assert output.shape == (1, 8, model_config.vocab_size)
        assert not torch.isnan(output).any()
        assert torch.isfinite(output).all()

    def test_model_checkpoint_loading(self):
        """Test saving and loading model checkpoints."""
        model_config = ModelConfigPresets.small_gpt(context_length=16)
        model = GPT(model_config)

        # Train briefly
        train_dataset = self.create_synthetic_dataset(
            vocab_size=model_config.vocab_size, seq_length=8, num_samples=10
        )
        val_dataset = self.create_synthetic_dataset(
            vocab_size=model_config.vocab_size, seq_length=8, num_samples=5
        )

        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

        with tempfile.TemporaryDirectory() as temp_dir:
            training_config = BaseTrainingConfig(
                num_epochs=1,
                eval_freq=5,
                eval_iter=1,
                learning_rate=1e-3,
                weight_decay=0.01,
                device='cpu',
                save_dir=temp_dir,
            )

            trainer = Trainer(model, train_loader, val_loader, training_config)
            trainer.train()

            # Save model
            trainer.save_model()
            save_path = (
                Path(temp_dir) / 'model_weights' / 'model_and_optimizer.pth'
            )

            # Create new model and load checkpoint
            new_model = GPT(model_config)
            checkpoint = torch.load(save_path, map_location='cpu')
            new_model.load_state_dict(checkpoint['model_state_dict'])

            # Test that loaded model produces same output
            test_input = torch.randint(0, model_config.vocab_size, (1, 8))

            model.eval()
            new_model.eval()

            with torch.no_grad():
                original_output = model(test_input)
                loaded_output = new_model(test_input)

            torch.testing.assert_close(original_output, loaded_output)

    def test_training_with_different_optimizers(self):
        """Test training workflow with different optimizer configurations."""
        model_config = ModelConfigPresets.small_gpt(context_length=16)
        model = GPT(model_config)

        train_dataset = self.create_synthetic_dataset(
            vocab_size=model_config.vocab_size, seq_length=8, num_samples=10
        )
        val_dataset = self.create_synthetic_dataset(
            vocab_size=model_config.vocab_size, seq_length=8, num_samples=5
        )

        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

        # Test different learning rates and weight decay values
        configs = [
            {'learning_rate': 1e-3, 'weight_decay': 0.01},
            {'learning_rate': 1e-4, 'weight_decay': 0.1},
            {'learning_rate': 1e-2, 'weight_decay': 0.0},
        ]

        for config in configs:
            # Reset model parameters
            model = GPT(model_config)

            training_config = BaseTrainingConfig(
                num_epochs=1,
                eval_freq=5,
                eval_iter=1,
                learning_rate=config['learning_rate'],
                weight_decay=config['weight_decay'],
                device='cpu',
            )

            trainer = Trainer(model, train_loader, val_loader, training_config)
            result = trainer.train()

            # Verify training completed without errors
            assert len(result['train_losses']) > 0
            assert len(result['val_losses']) > 0

    def test_training_with_tokenizer_integration(self):
        """Test training workflow integrated with tokenizers."""
        model_config = ModelConfigPresets.small_gpt(context_length=16)
        model = GPT(model_config)

        # Test with BPE tokenizer
        bpe_tokenizer = BPETokenizer()

        # Create text data and tokenize
        sample_texts = [
            'Hello world, this is a test.',
            'Machine learning is fascinating.',
            'Natural language processing with transformers.',
            'Training neural networks requires patience.',
        ]

        # Tokenize texts
        tokenized_inputs = []
        for text in sample_texts:
            tokens = bpe_tokenizer.encode(text)
            # Pad or truncate to fixed length
            if len(tokens) >= 8:
                tokens = tokens[:8]
            else:
                tokens = tokens + [0] * (8 - len(tokens))  # Pad with 0s
            tokenized_inputs.append(tokens)

        # Create dataset from tokenized text
        inputs = torch.tensor(tokenized_inputs)
        targets = inputs.clone()  # For language modeling, targets = inputs

        train_dataset = TensorDataset(inputs, targets)
        val_dataset = TensorDataset(inputs[:2], targets[:2])

        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

        training_config = BaseTrainingConfig(
            num_epochs=1,
            eval_freq=1,
            eval_iter=1,
            learning_rate=1e-4,
            weight_decay=0.01,
            device='cpu',
        )

        trainer = Trainer(
            model, train_loader, val_loader, training_config, bpe_tokenizer
        )
        result = trainer.train()

        # Verify training completed
        assert len(result['train_losses']) > 0
        assert trainer.tokenizer is bpe_tokenizer

    def test_gradient_accumulation_simulation(self):
        """Test simulated gradient accumulation for larger effective batch sizes."""
        model_config = ModelConfigPresets.small_gpt(context_length=16)
        model = GPT(model_config)

        train_dataset = self.create_synthetic_dataset(
            vocab_size=model_config.vocab_size, seq_length=8, num_samples=16
        )
        val_dataset = self.create_synthetic_dataset(
            vocab_size=model_config.vocab_size, seq_length=8, num_samples=4
        )

        # Small batch size to simulate gradient accumulation
        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

        training_config = BaseTrainingConfig(
            num_epochs=1,
            eval_freq=10,
            eval_iter=1,
            learning_rate=1e-3,
            weight_decay=0.01,
            device='cpu',
        )

        trainer = Trainer(model, train_loader, val_loader, training_config)

        # Simulate gradient accumulation
        accumulation_steps = 2

        model.train()
        for _ in range(training_config.num_epochs):
            for i, (input_batch, target_batch) in enumerate(train_loader):
                loss = trainer.train_step(input_batch, target_batch)
                loss = loss / accumulation_steps  # Scale loss
                loss.backward()

                if (i + 1) % accumulation_steps == 0:
                    trainer.optimizer.step()
                    trainer.optimizer.zero_grad()

        # Verify training metrics were updated
        assert trainer.global_step > 0
        assert trainer.examples_seen > 0

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason='CUDA not available'
    )
    def test_gpu_training_workflow(self):
        """Test training workflow on GPU (if available)."""
        model_config = ModelConfigPresets.small_gpt(context_length=16)
        model = GPT(model_config)

        train_dataset = self.create_synthetic_dataset(
            vocab_size=model_config.vocab_size, seq_length=8, num_samples=10
        )
        val_dataset = self.create_synthetic_dataset(
            vocab_size=model_config.vocab_size, seq_length=8, num_samples=5
        )

        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

        training_config = BaseTrainingConfig(
            num_epochs=1,
            eval_freq=5,
            eval_iter=1,
            learning_rate=1e-3,
            weight_decay=0.01,
            device='cuda',
        )

        trainer = Trainer(model, train_loader, val_loader, training_config)

        # Verify model is on GPU
        assert next(trainer.model.parameters()).device.type == 'cuda'

        # Run training
        result = trainer.train()

        # Verify training completed
        assert len(result['train_losses']) > 0
        assert len(result['val_losses']) > 0

    def test_overfitting_small_dataset(self):
        """Test that model can overfit to a very small dataset."""
        model_config = ModelConfigPresets.small_gpt(context_length=16)
        model = GPT(model_config)

        # Create tiny dataset that model should be able to memorize
        train_dataset = self.create_synthetic_dataset(
            vocab_size=100,  # Small vocab
            seq_length=4,  # Short sequences
            num_samples=4,  # Very few samples
        )

        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
        val_loader = DataLoader(
            train_dataset, batch_size=1, shuffle=False
        )  # Same as train

        training_config = BaseTrainingConfig(
            num_epochs=10,  # More epochs to allow overfitting
            eval_freq=5,
            eval_iter=4,  # Evaluate on all samples
            learning_rate=1e-2,  # Higher learning rate
            weight_decay=0.0,  # No regularization
            device='cpu',
        )

        trainer = Trainer(model, train_loader, val_loader, training_config)
        result = trainer.train()

        # Check that loss decreased significantly
        initial_loss = result['train_losses'][0]
        final_loss = result['train_losses'][-1]

        assert final_loss < initial_loss, (
            'Loss should decrease with overfitting'
        )


class TestMoETrainingIntegration:
    """Integration tests for MoE model training with aux_loss."""

    def _make_config(self, **kwargs: object) -> ModelConfig:
        """Create a small config for testing."""
        defaults = {
            'vocab_size': 256,
            'embed_dim': 64,
            'num_heads': 4,
            'num_kv_heads': 4,
            'num_layers': 2,
            'context_length': 32,
            'dropout': 0.0,
        }
        defaults.update(kwargs)
        return ModelConfig(**defaults)

    def test_mixtral_training_with_aux_loss(self) -> None:
        """Mixtral training loop includes load balancing loss."""
        from lmt.models.mixtral import Mixtral

        config = self._make_config()
        model = Mixtral(config, num_experts=4, top_k=2)

        inputs = torch.randint(0, 256, (12, 16))
        targets = torch.randint(0, 256, (12, 16))
        dataset = TensorDataset(inputs, targets)
        loader = DataLoader(dataset, batch_size=4)

        training_config = BaseTrainingConfig(
            num_epochs=2,
            eval_freq=1,
            eval_iter=1,
            learning_rate=1e-3,
            weight_decay=0.0,
            device='cpu',
            aux_loss_coeff=0.01,
        )

        trainer = Trainer(model, loader, loader, training_config)
        result = trainer.train()

        assert len(result['train_losses']) > 0
        # aux_loss should be non-negative after forward pass
        assert model.aux_loss.item() >= 0.0

    def test_deepseek_training_with_aux_loss(self) -> None:
        """DeepSeek-V2 training loop includes load balancing loss."""
        from lmt.models.deepseek import DeepSeekV2

        config = self._make_config()
        model = DeepSeekV2(
            config,
            kv_compress_dim=32,
            q_compress_dim=32,
            rope_dim=8,
            num_experts=4,
            top_k=2,
        )

        inputs = torch.randint(0, 256, (12, 16))
        targets = torch.randint(0, 256, (12, 16))
        dataset = TensorDataset(inputs, targets)
        loader = DataLoader(dataset, batch_size=4)

        training_config = BaseTrainingConfig(
            num_epochs=2,
            eval_freq=1,
            eval_iter=1,
            learning_rate=1e-3,
            weight_decay=0.0,
            device='cpu',
            aux_loss_coeff=0.01,
        )

        trainer = Trainer(model, loader, loader, training_config)
        result = trainer.train()

        assert len(result['train_losses']) > 0
        assert model.aux_loss.item() >= 0.0

    def test_moe_loss_decreases_during_training(self) -> None:
        """MoE model loss decreases when training with aux_loss."""
        from lmt.models.mixtral import Mixtral

        config = self._make_config()
        model = Mixtral(config, num_experts=4, top_k=2)

        # Small repeating dataset to encourage overfitting
        inputs = torch.randint(0, 256, (4, 8))
        targets = torch.randint(0, 256, (4, 8))
        dataset = TensorDataset(inputs, targets)
        loader = DataLoader(dataset, batch_size=2, shuffle=False)

        training_config = BaseTrainingConfig(
            num_epochs=10,
            eval_freq=5,
            eval_iter=2,
            learning_rate=1e-2,
            weight_decay=0.0,
            device='cpu',
            aux_loss_coeff=0.01,
        )

        trainer = Trainer(model, loader, loader, training_config)
        result = trainer.train()

        initial_loss = result['train_losses'][0]
        final_loss = result['train_losses'][-1]
        assert final_loss < initial_loss, (
            f'MoE loss should decrease: {initial_loss:.3f} -> {final_loss:.3f}'
        )
