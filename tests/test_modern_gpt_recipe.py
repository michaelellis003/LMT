"""Integration tests for the modern GPT pretraining recipe.

Validates the full pipeline: model creation with modern tricks,
training via ExperimentRunner, and BPB evaluation.
"""

import torch

from lmt.data.token_dataset import TokenDataset
from lmt.eval.bpb import compute_bpb
from lmt.models.config import ModelConfig
from lmt.research.experiment import ExperimentRunConfig, ExperimentRunner
from lmt.training.reproducibility import set_seed


def _tiny_config() -> ModelConfig:
    """Create a tiny model config for fast testing."""
    return ModelConfig(
        embed_dim=64,
        num_heads=4,
        num_kv_heads=2,
        num_layers=2,
        ffn_hidden_dim=128,
        vocab_size=256,
        context_length=32,
        tie_weights=True,
        qk_norm=True,
        dropout=0.0,
    )


class TestModernGPTRecipe:
    """Integration tests for modern GPT recipe components."""

    def test_qwen3_with_modern_tricks(self):
        """Qwen3 model with QK-norm and tied weights should train."""
        from lmt.models.qwen3.qwen3 import Qwen3

        set_seed(42)
        config = _tiny_config()
        model = Qwen3(config)

        tokens = torch.randint(0, config.vocab_size, (500,))
        dataset = TokenDataset(tokens, config.context_length)
        train_data = torch.stack([dataset[i] for i in range(len(dataset))])

        exp_config = ExperimentRunConfig(
            name='test_modern_gpt',
            train_steps=5,
            eval_interval=5,
            lr=1e-3,
            batch_size=4,
        )

        runner = ExperimentRunner(output_dir='/tmp/test_modern_gpt')
        result = runner.run(model, train_data, exp_config)

        assert result.final_loss > 0
        assert result.final_loss < 10  # sanity: not diverged

    def test_bpb_evaluation(self):
        """BPB eval should work on a model trained via recipe."""
        from lmt.models.qwen3.qwen3 import Qwen3

        set_seed(42)
        config = _tiny_config()
        model = Qwen3(config)

        seq_len = config.context_length + 1
        eval_tokens = torch.randint(0, config.vocab_size, (3, seq_len))
        bpb = compute_bpb(model, eval_tokens, bytes_per_token=1.0)

        assert isinstance(bpb, float)
        assert bpb > 0

    def test_relu_squared_as_ffn(self):
        """ReLU² should work as a drop-in SwiGLU replacement."""
        from lmt.layers.blocks.configurable_block import BlockConfig
        from lmt.models.base import BaseModel

        config = _tiny_config()
        bc = BlockConfig(ffn='relu_squared')
        model = BaseModel(config, block_config=bc)

        x = torch.randint(0, config.vocab_size, (2, 16))
        logits = model(x)
        assert logits.shape == (2, 16, config.vocab_size)

    def test_sliding_window_attention(self):
        """Sliding window GQA should work in the model."""
        from lmt.layers.blocks.configurable_block import BlockConfig
        from lmt.layers.positional.rope import RoPE
        from lmt.models.base import BaseModel

        config = _tiny_config()
        config.window_size = 8
        rope = RoPE(
            config.embed_dim // config.num_heads, config.context_length
        )
        bc = BlockConfig(attention='gqa', rope=rope, window_size=8)
        model = BaseModel(config, block_config=bc)

        x = torch.randint(0, config.vocab_size, (2, 16))
        logits = model(x)
        assert logits.shape == (2, 16, config.vocab_size)

    def test_loss_decreases(self):
        """Loss should decrease over training steps."""
        from lmt.models.qwen3.qwen3 import Qwen3

        set_seed(42)
        config = _tiny_config()
        model = Qwen3(config)

        tokens = torch.randint(0, config.vocab_size, (1000,))
        dataset = TokenDataset(tokens, config.context_length)
        train_data = torch.stack([dataset[i] for i in range(len(dataset))])

        exp_config = ExperimentRunConfig(
            name='test_loss_decrease',
            train_steps=20,
            eval_interval=10,
            lr=1e-3,
            batch_size=8,
        )

        runner = ExperimentRunner(output_dir='/tmp/test_loss_decrease')
        result = runner.run(model, train_data, exp_config)

        # Loss should decrease over 20 steps on random data
        loss_history = result.metrics.get_history('train_loss')
        first_loss = loss_history[0][1]  # (step, value)
        assert result.final_loss < first_loss

    def test_muon_optimizer(self):
        """Muon optimizer should work with the training pipeline."""
        from lmt.models.qwen3.qwen3 import Qwen3
        from lmt.training.muon import Muon

        set_seed(42)
        config = _tiny_config()
        model = Qwen3(config)

        tokens = torch.randint(0, config.vocab_size, (500,))
        dataset = TokenDataset(tokens, config.context_length)
        train_data = torch.stack([dataset[i] for i in range(len(dataset))])

        # Manual training loop with Muon
        inputs = train_data[:8, :-1]
        targets = train_data[:8, 1:]

        optimizer = Muon(model.parameters(), lr=1e-3)
        model.train()
        logits = model(inputs)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, config.vocab_size), targets.reshape(-1)
        )
        loss.backward()
        optimizer.step()

        assert loss.item() > 0
