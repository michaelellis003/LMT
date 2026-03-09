"""End-to-end integration test for the autoresearch framework.

Exercises the full loop: create model → train → evaluate BPB →
compare experiments → identify best. This validates that all
research infrastructure modules work together correctly.
"""

from pathlib import Path

import torch

from lmt.data.token_dataset import TokenDataset
from lmt.eval.bpb import compute_bpb
from lmt.models.config import ModelConfig
from lmt.research.experiment import (
    ExperimentRunConfig,
    ExperimentRunner,
)
from lmt.research.search import SearchSpace, grid_search
from lmt.training.reproducibility import set_seed


def _tiny_config() -> ModelConfig:
    """Create a tiny model config."""
    return ModelConfig(
        embed_dim=32,
        num_heads=2,
        num_kv_heads=2,
        num_layers=1,
        ffn_hidden_dim=64,
        vocab_size=64,
        context_length=16,
        tie_weights=True,
        dropout=0.0,
    )


class TestAutoresearchLoop:
    """Integration test for the full autoresearch pipeline."""

    def test_full_loop(self, tmp_path: Path):
        """Should train, evaluate BPB, and compare experiments."""
        from lmt.models.qwen3.qwen3 import Qwen3

        set_seed(42)

        # 1. Create training and validation data
        train_tokens = torch.randint(0, 64, (200,))
        val_tokens = torch.randint(0, 64, (100,))
        train_data = TokenDataset(train_tokens, context_length=16)

        # Convert to batched tensors for experiment runner
        train_batch = torch.stack(
            [train_data[i] for i in range(len(train_data))]
        )
        val_batch = torch.stack(
            [
                val_tokens[i * 17 : i * 17 + 17]
                for i in range(len(val_tokens) // 17)
            ]
        )

        # 2. Run grid search over learning rates
        config = _tiny_config()

        def model_fn():
            set_seed(42)
            return Qwen3(config)

        space = SearchSpace(lr=[1e-2, 1e-3])
        base_config = ExperimentRunConfig(
            name='autoresearch',
            train_steps=10,
            eval_interval=10,
            batch_size=2,
        )

        results = grid_search(
            model_fn=model_fn,
            train_data=train_batch,
            search_space=space,
            base_config=base_config,
            output_dir=str(tmp_path / 'grid'),
            val_data=val_batch,
        )

        # 3. Verify results
        assert len(results) == 2

        for result in results:
            # Should have training loss
            train_history = result.metrics.get_history('train_loss')
            assert len(train_history) > 0

            # Should have val_bpb
            bpb_history = result.metrics.get_history('val_bpb')
            assert len(bpb_history) >= 1
            assert bpb_history[-1][1] > 0

            # Final loss should be recorded
            assert result.final_loss is not None

        # 4. Compare and identify best
        bpbs = [r.metrics.get_best('val_bpb') for r in results]
        assert all(b is not None for b in bpbs)
        best_idx = bpbs.index(min(b for b in bpbs if b is not None))
        best = results[best_idx]
        assert best.config.lr in [1e-2, 1e-3]

    def test_experiment_runner_standalone(self, tmp_path: Path):
        """Should work with a single experiment run."""
        from lmt.models.qwen3.qwen3 import Qwen3

        set_seed(42)
        model = Qwen3(_tiny_config())
        train_data = torch.randint(0, 64, (10, 17))
        val_data = torch.randint(0, 64, (5, 17))

        config = ExperimentRunConfig(
            name='single',
            train_steps=5,
            eval_interval=5,
            lr=1e-3,
            batch_size=2,
        )

        runner = ExperimentRunner(output_dir=str(tmp_path))
        result = runner.run(model, train_data, config, val_data=val_data)

        # Verify full result structure
        assert result.config.name == 'single'
        assert result.final_loss is not None
        assert result.metrics.get_best('val_bpb') is not None

        # Verify saved to disk
        result_file = tmp_path / 'single' / 'result.json'
        assert result_file.exists()

    def test_bpb_standalone(self):
        """BPB should work independently on a tiny model."""
        from lmt.models.qwen3.qwen3 import Qwen3

        set_seed(42)
        model = Qwen3(_tiny_config())
        model.eval()

        tokens = torch.randint(0, 64, (3, 17))
        bpb = compute_bpb(model, tokens, bytes_per_token=3.5)

        assert isinstance(bpb, float)
        assert bpb > 0
        # Untrained model should have high BPB
        assert bpb > 1.0

    def test_training_improves_bpb(self, tmp_path: Path):
        """Training should reduce BPB on memorizable data."""
        from lmt.models.qwen3.qwen3 import Qwen3

        set_seed(42)
        config = _tiny_config()
        model = Qwen3(config)

        # Create repeatable data (easier to memorize)
        data = torch.randint(0, 64, (4, 17))
        data = data.repeat(5, 1)  # Repeat for memorization

        # Measure BPB before training
        model.eval()
        bpb_before = compute_bpb(model, data[:4], bytes_per_token=1.0)

        # Train
        exp_config = ExperimentRunConfig(
            name='memorize',
            train_steps=30,
            eval_interval=30,
            lr=1e-2,
            batch_size=4,
        )

        runner = ExperimentRunner(output_dir=str(tmp_path))
        runner.run(model, data, exp_config)

        # Measure BPB after training
        model.eval()
        bpb_after = compute_bpb(model, data[:4], bytes_per_token=1.0)

        # BPB should decrease after training
        assert bpb_after < bpb_before
