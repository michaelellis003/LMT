"""Integration tests for the research pipeline.

Validates that all research infrastructure components work together:
model creation → training → BPB evaluation → HF export → reload.

This is the smoke test for the entire autoresearch pipeline. If this
passes, all the individual pieces are correctly integrated.
"""

from pathlib import Path

import torch

from lmt.models.config import ModelConfig


def _tiny_config() -> ModelConfig:
    """Create a minimal Qwen3-like config for integration testing."""
    return ModelConfig(
        embed_dim=32,
        num_heads=4,
        num_kv_heads=2,
        num_layers=2,
        ffn_hidden_dim=64,
        vocab_size=64,
        context_length=16,
        tie_weights=True,
        qk_norm=True,
        dropout=0.0,
    )


class TestExperimentRunnerPipeline:
    """Test: experiment runner → metrics → checkpoint."""

    def test_train_and_evaluate(self, tmp_path: Path):
        """Train a model with experiment runner, verify loss drops."""
        from lmt.models.qwen3.qwen3 import Qwen3
        from lmt.research.experiment import (
            ExperimentRunConfig,
            ExperimentRunner,
        )

        config = _tiny_config()
        model = Qwen3(config)

        # Create simple training data (repeated for memorization)
        data = torch.randint(0, 64, (8, 12)).repeat(5, 1)

        exp_config = ExperimentRunConfig(
            name='integration_test',
            train_steps=20,
            eval_interval=10,
            lr=5e-3,
            batch_size=4,
            save_checkpoint=True,
        )

        runner = ExperimentRunner(output_dir=str(tmp_path))
        result = runner.run(model, data, exp_config)

        # Verify training happened
        assert result.final_loss is not None
        assert isinstance(result.final_loss, float)

        # Verify loss decreased
        history = result.metrics.get_history('train_loss')
        assert len(history) == 20
        assert history[-1][1] < history[0][1]

        # Verify checkpoint was saved
        assert (tmp_path / 'integration_test' / 'checkpoint.pt').exists()
        assert (tmp_path / 'integration_test' / 'result.json').exists()


class TestBPBPipeline:
    """Test: model → BPB evaluation."""

    def test_bpb_on_trained_model(self, tmp_path: Path):
        """Compute BPB on a model after training."""
        from lmt.eval.bpb import compute_bpb
        from lmt.models.qwen3.qwen3 import Qwen3
        from lmt.research.experiment import (
            ExperimentRunConfig,
            ExperimentRunner,
        )

        config = _tiny_config()
        model = Qwen3(config)

        # Train briefly
        data = torch.randint(0, 64, (8, 12)).repeat(5, 1)
        exp_config = ExperimentRunConfig(
            name='bpb_test',
            train_steps=10,
            lr=5e-3,
        )
        runner = ExperimentRunner(output_dir=str(tmp_path))
        runner.run(model, data, exp_config)

        # Evaluate BPB
        eval_data = torch.randint(0, 64, (4, 12))
        bpb = compute_bpb(model, eval_data, bytes_per_token=1.0)

        assert isinstance(bpb, float)
        assert bpb > 0
        assert not torch.isnan(torch.tensor(bpb))


class TestExportPipeline:
    """Test: train → export → reload → verify."""

    def test_train_export_reload(self, tmp_path: Path):
        """Full pipeline: train, export to HF, reload, verify outputs."""
        from lmt.models.hf_exporter import export_to_hf
        from lmt.models.hf_loader import load_hf_state_dict
        from lmt.models.qwen3.qwen3 import Qwen3
        from lmt.research.experiment import (
            ExperimentRunConfig,
            ExperimentRunner,
        )

        config = _tiny_config()
        model = Qwen3(config)

        # Train
        data = torch.randint(0, 64, (8, 12)).repeat(5, 1)
        exp_config = ExperimentRunConfig(
            name='export_test',
            train_steps=5,
            lr=5e-3,
        )
        runner = ExperimentRunner(output_dir=str(tmp_path))
        runner.run(model, data, exp_config)

        # Export to HF format
        export_dir = tmp_path / 'hf_export'
        export_to_hf(model, str(export_dir), 'qwen3')

        assert (export_dir / 'model.safetensors').exists()
        assert (export_dir / 'config.json').exists()

        # Reload into fresh model
        from safetensors.torch import load_file

        hf_state = load_file(str(export_dir / 'model.safetensors'))
        restored = Qwen3(config)
        load_hf_state_dict(restored, hf_state, 'qwen3')

        # Verify outputs match
        model.eval()
        restored.eval()
        test_input = torch.randint(0, 64, (1, 8))
        with torch.no_grad():
            orig_out = model(test_input)
            rest_out = restored(test_input)

        assert torch.allclose(orig_out, rest_out, atol=1e-6)


class TestReproducibilityPipeline:
    """Test: seeded training produces identical results."""

    def test_reproducible_training(self, tmp_path: Path):
        """Same seed + config should produce identical final loss."""
        from lmt.models.qwen3.qwen3 import Qwen3
        from lmt.research.experiment import (
            ExperimentRunConfig,
            ExperimentRunner,
        )
        from lmt.training.reproducibility import set_seed

        config = _tiny_config()
        data = torch.randint(0, 64, (8, 12))

        # Run 1
        set_seed(42)
        model1 = Qwen3(config)
        exp_config = ExperimentRunConfig(
            name='repro_1',
            train_steps=10,
            lr=5e-3,
            batch_size=4,
        )
        runner1 = ExperimentRunner(output_dir=str(tmp_path / 'run1'))
        result1 = runner1.run(model1, data, exp_config)

        # Run 2 (same seed)
        set_seed(42)
        model2 = Qwen3(config)
        exp_config2 = ExperimentRunConfig(
            name='repro_2',
            train_steps=10,
            lr=5e-3,
            batch_size=4,
        )
        runner2 = ExperimentRunner(output_dir=str(tmp_path / 'run2'))
        result2 = runner2.run(model2, data, exp_config2)

        # Same seed → same result
        assert result1.final_loss == result2.final_loss


class TestRewardIntegration:
    """Test: reward functions work with data pipeline."""

    def test_math_pipeline(self):
        """Math data → format prompt → generate answer → score."""
        from lmt.data.math_data import MathDataItem, format_math_prompt
        from lmt.training.rewards import math_reward

        item = MathDataItem(
            prompt='What is 2 + 3?',
            ground_truth='5',
        )

        # Format prompt
        formatted = format_math_prompt(item.prompt)
        assert item.prompt in formatted

        # Simulate a correct model response
        response = r'2 + 3 = 5. The answer is \boxed{5}'
        reward = math_reward(
            prompt=formatted,
            response=response,
            ground_truth=item.ground_truth,
        )
        assert reward == 1.0

        # Simulate an incorrect response
        wrong_response = r'2 + 3 = 6. The answer is \boxed{6}'
        reward = math_reward(
            prompt=formatted,
            response=wrong_response,
            ground_truth=item.ground_truth,
        )
        assert reward == 0.0

    def test_gsm8k_pipeline(self):
        """GSM8K answer extraction → reward scoring."""
        from lmt.data.math_data import MathDataItem, extract_gsm8k_answer
        from lmt.training.rewards import math_reward

        # Simulate GSM8K format
        solution = 'John has 5 apples. He gives 2 away. 5 - 2 = 3\n#### 3'
        answer = extract_gsm8k_answer(solution)
        assert answer == '3'

        # Create a data item and score a model response
        item = MathDataItem(
            prompt='How many apples does John have?',
            ground_truth=answer,
        )

        response = r'The answer is \boxed{3}'
        reward = math_reward(
            prompt=item.prompt,
            response=response,
            ground_truth=item.ground_truth,
        )
        assert reward == 1.0
