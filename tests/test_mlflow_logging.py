"""Tests for MLflow experiment tracking integration."""

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


class TestMLflowConfig:
    """Test MLflow configuration in BaseTrainingConfig."""

    def test_config_has_run_name(self) -> None:
        """run_name field exists for naming MLflow runs."""
        config = BaseTrainingConfig(
            num_epochs=1,
            eval_freq=10,
            eval_iter=1,
            learning_rate=1e-3,
            weight_decay=0.0,
            run_name='my-run',
        )
        assert config.run_name == 'my-run'

    def test_run_name_defaults_to_none(self) -> None:
        """run_name defaults to None (MLflow disabled)."""
        config = BaseTrainingConfig(
            num_epochs=1,
            eval_freq=10,
            eval_iter=1,
            learning_rate=1e-3,
            weight_decay=0.0,
        )
        assert config.run_name is None


class TestMLflowLogging:
    """Test MLflow integration in Trainer."""

    def test_mlflow_creates_run_dir(self) -> None:
        """Trainer creates MLflow run artifacts when run_name is set."""
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

            # MLflow creates an mlruns directory
            mlruns_dir = Path(tmp) / 'mlruns'
            assert mlruns_dir.exists(), f'MLflow dir {mlruns_dir} should exist'

    def test_mlflow_disabled_by_default(self) -> None:
        """MLflow is disabled when no run_name is set."""
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

            mlruns_dir = Path(tmp) / 'mlruns'
            assert not mlruns_dir.exists(), (
                'No mlruns dir when run_name not set'
            )

    def test_mlflow_logs_metrics(self) -> None:
        """MLflow run contains logged metrics."""
        import mlflow

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
                run_name='metric_test',
            )
            trainer = Trainer(model, loader, loader, config)
            trainer.train()

            # Query MLflow for the run
            tracking_uri = f'file://{Path(tmp).resolve()}/mlruns'
            client = mlflow.tracking.MlflowClient(tracking_uri)
            experiments = client.search_experiments()
            assert len(experiments) > 0

            runs = client.search_runs(
                experiment_ids=[experiments[0].experiment_id]
            )
            assert len(runs) == 1
            run = runs[0]

            # Check that metrics were logged
            assert 'train_step_loss' in run.data.metrics
            assert 'train_loss' in run.data.metrics
            assert 'val_loss' in run.data.metrics

    def test_mlflow_logs_params(self) -> None:
        """MLflow run contains training config as params."""
        import mlflow

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
                run_name='param_test',
            )
            trainer = Trainer(model, loader, loader, config)
            trainer.train()

            tracking_uri = f'file://{Path(tmp).resolve()}/mlruns'
            client = mlflow.tracking.MlflowClient(tracking_uri)
            experiments = client.search_experiments()
            runs = client.search_runs(
                experiment_ids=[experiments[0].experiment_id]
            )
            run = runs[0]

            assert run.data.params['learning_rate'] == '0.001'
            assert run.data.params['num_epochs'] == '1'
            assert 'param_count' in run.data.params

    def test_mlflow_with_moe_model(self) -> None:
        """MLflow works with MoE models that have aux_loss."""
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

            mlruns_dir = Path(tmp) / 'mlruns'
            assert mlruns_dir.exists()

    def test_return_dict_unchanged(self) -> None:
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

            assert 'train_losses' in result
            assert 'val_losses' in result
            assert 'execution_time' in result
            assert len(result['train_losses']) > 0
