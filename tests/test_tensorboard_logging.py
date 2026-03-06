"""Tests for TensorBoard experiment tracking integration."""

import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from lmt.models.config import ModelConfig, ModelConfigPresets
from lmt.models.gpt import GPT
from lmt.training.config import BaseTrainingConfig
from lmt.training.trainer import Trainer


def _make_loader(
    vocab_size: int = 256, seq_len: int = 8, n: int = 12, batch: int = 4
) -> DataLoader:
    """Create a small dataloader for testing."""
    inputs = torch.randint(0, vocab_size, (n, seq_len))
    targets = torch.randint(0, vocab_size, (n, seq_len))
    return DataLoader(TensorDataset(inputs, targets), batch_size=batch)


class TestTensorBoardLogging:
    """Test TensorBoard integration in Trainer."""

    def test_tensorboard_creates_log_dir(self) -> None:
        """Trainer creates TensorBoard log directory."""
        model = GPT(ModelConfigPresets.small_gpt(context_length=16))
        loader = _make_loader(vocab_size=model.tok_embed.num_embeddings)

        with tempfile.TemporaryDirectory() as tmp:
            config = BaseTrainingConfig(
                num_epochs=1,
                eval_freq=1,
                eval_iter=1,
                learning_rate=1e-3,
                weight_decay=0.0,
                device='cpu',
                save_dir=tmp,
                run_name='test_run',
            )
            trainer = Trainer(model, loader, loader, config)
            trainer.train()

            # TensorBoard log directory should exist
            tb_dir = Path(tmp) / 'tb_logs' / 'test_run'
            assert tb_dir.exists(), f'TensorBoard dir {tb_dir} should exist'

            # Should contain event files
            event_files = list(tb_dir.glob('events.out.tfevents.*'))
            assert len(event_files) > 0, 'Should have event files'

    def test_tensorboard_disabled_by_default(self) -> None:
        """TensorBoard is disabled when no run_name is set."""
        model = GPT(ModelConfigPresets.small_gpt(context_length=16))
        loader = _make_loader(vocab_size=model.tok_embed.num_embeddings)

        with tempfile.TemporaryDirectory() as tmp:
            config = BaseTrainingConfig(
                num_epochs=1,
                eval_freq=1,
                eval_iter=1,
                learning_rate=1e-3,
                weight_decay=0.0,
                device='cpu',
                save_dir=tmp,
            )
            trainer = Trainer(model, loader, loader, config)
            trainer.train()

            # No tb_logs directory should be created
            tb_dir = Path(tmp) / 'tb_logs'
            assert not tb_dir.exists(), 'No TB dir when run_name not set'

    def test_tensorboard_logs_scalars(self) -> None:
        """TensorBoard logs contain loss scalars."""
        model = GPT(ModelConfigPresets.small_gpt(context_length=16))
        loader = _make_loader(vocab_size=model.tok_embed.num_embeddings)

        with tempfile.TemporaryDirectory() as tmp:
            config = BaseTrainingConfig(
                num_epochs=1,
                eval_freq=1,
                eval_iter=1,
                learning_rate=1e-3,
                weight_decay=0.0,
                device='cpu',
                save_dir=tmp,
                run_name='scalar_test',
            )
            trainer = Trainer(model, loader, loader, config)
            trainer.train()

            # Event files should have nonzero size (contain data)
            tb_dir = Path(tmp) / 'tb_logs' / 'scalar_test'
            event_files = list(tb_dir.glob('events.out.tfevents.*'))
            total_size = sum(f.stat().st_size for f in event_files)
            assert total_size > 100, 'Event files should contain logged data'

    def test_tensorboard_with_moe_model(self) -> None:
        """TensorBoard works with MoE models that have aux_loss."""
        from lmt.models.mixtral import Mixtral

        config = ModelConfig(
            vocab_size=256,
            embed_dim=64,
            num_heads=4,
            num_kv_heads=4,
            num_layers=2,
            context_length=32,
            dropout=0.0,
        )
        model = Mixtral(config, num_experts=4, top_k=2)
        loader = _make_loader()

        with tempfile.TemporaryDirectory() as tmp:
            train_config = BaseTrainingConfig(
                num_epochs=1,
                eval_freq=1,
                eval_iter=1,
                learning_rate=1e-3,
                weight_decay=0.0,
                device='cpu',
                save_dir=tmp,
                run_name='moe_test',
                aux_loss_coeff=0.01,
            )
            trainer = Trainer(model, loader, loader, train_config)
            trainer.train()

            tb_dir = Path(tmp) / 'tb_logs' / 'moe_test'
            assert tb_dir.exists()

    def test_existing_behavior_unchanged(self) -> None:
        """Trainer still returns the same results dict."""
        model = GPT(ModelConfigPresets.small_gpt(context_length=16))
        loader = _make_loader(vocab_size=model.tok_embed.num_embeddings)

        with tempfile.TemporaryDirectory() as tmp:
            config = BaseTrainingConfig(
                num_epochs=1,
                eval_freq=1,
                eval_iter=1,
                learning_rate=1e-3,
                weight_decay=0.0,
                device='cpu',
                save_dir=tmp,
                run_name='compat_test',
            )
            trainer = Trainer(model, loader, loader, config)
            result = trainer.train()

            # Original return values still present
            assert 'train_losses' in result
            assert 'val_losses' in result
            assert 'execution_time' in result
            assert len(result['train_losses']) > 0
