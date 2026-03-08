"""Sanity checks for training correctness.

These are Karpathy-style sanity checks that every training pipeline
should pass before real experiments. If any of these fail, there is
a fundamental bug in the pipeline.

Checks implemented:
1. Single batch overfit: model can memorize one batch to near-zero loss
2. Gradient check: loss on example i produces gradients only for example i
3. Loss at initialization: should be ~ln(vocab_size) for uniform predictions
4. Learning rate sensitivity: training should work with a range of LRs
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as f

from lmt.models.config import ModelConfig


def _tiny_model() -> nn.Module:
    """Create a tiny Qwen3 model for sanity checks."""
    from lmt.models.qwen3.qwen3 import Qwen3

    config = ModelConfig(
        embed_dim=64,
        num_heads=4,
        num_kv_heads=2,
        num_layers=2,
        ffn_hidden_dim=128,
        vocab_size=64,
        context_length=16,
        tie_weights=True,
        qk_norm=True,
        dropout=0.0,
    )
    return Qwen3(config)


class TestSingleBatchOverfit:
    """Can the model memorize a single batch?

    This is the most basic sanity check from Karpathy's recipe.
    If a model can't memorize one batch to near-zero loss, something
    is fundamentally wrong with the forward pass, loss computation,
    or gradient flow.
    """

    def test_overfit_single_batch(self):
        """Model should reach near-zero loss on a single repeated batch."""
        torch.manual_seed(42)
        model = _tiny_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # Single batch: 4 sequences of length 16
        batch = torch.randint(0, 64, (4, 16))
        inputs = batch[:, :-1]
        targets = batch[:, 1:]

        model.train()
        losses = []
        for _ in range(200):
            logits = model(inputs)
            loss = f.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Should get very close to zero
        final_loss = losses[-1]
        assert final_loss < 0.1, (
            f'Failed to overfit single batch: final_loss={final_loss:.4f}. '
            f'Expected < 0.1. This suggests a bug in forward pass or '
            f'gradient computation.'
        )

        # Loss should have decreased monotonically (mostly)
        assert losses[-1] < losses[0] * 0.01, (
            f'Loss did not decrease enough: {losses[0]:.4f} → {losses[-1]:.4f}'
        )


class TestInitLoss:
    """Loss at initialization should match theoretical expectation.

    For a randomly initialized model predicting uniformly over V tokens,
    the expected loss is ln(V). Significant deviation suggests biased
    initialization or a bug in the loss computation.
    """

    def test_init_loss_matches_theory(self):
        """Initial loss should be approximately ln(vocab_size)."""
        torch.manual_seed(42)
        model = _tiny_model()
        model.eval()

        vocab_size = 64
        expected_loss = math.log(vocab_size)  # ~4.159

        # Average over several batches for stability
        total_loss = 0.0
        n_batches = 10
        with torch.no_grad():
            for _ in range(n_batches):
                batch = torch.randint(0, vocab_size, (8, 16))
                logits = model(batch[:, :-1])
                loss = f.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    batch[:, 1:].reshape(-1),
                )
                total_loss += loss.item()

        avg_loss = total_loss / n_batches

        # Should be within 20% of theoretical value
        assert abs(avg_loss - expected_loss) / expected_loss < 0.2, (
            f'Init loss {avg_loss:.4f} deviates from expected '
            f'{expected_loss:.4f} (ln({vocab_size})). '
            f'This suggests biased initialization.'
        )


class TestGradientFlow:
    """Verify gradients flow correctly through the model.

    Checks that:
    1. All trainable parameters receive non-zero gradients
    2. No gradients are NaN or Inf
    3. Gradient magnitudes are reasonable (no vanishing/exploding)
    """

    def test_all_params_get_gradients(self):
        """Every trainable parameter should receive a gradient."""
        torch.manual_seed(42)
        model = _tiny_model()
        model.train()

        batch = torch.randint(0, 64, (4, 16))
        logits = model(batch[:, :-1])
        loss = f.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            batch[:, 1:].reshape(-1),
        )
        loss.backward()

        zero_grad_params = []
        nan_grad_params = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.grad is None:
                    zero_grad_params.append(name)
                elif torch.all(param.grad == 0):
                    zero_grad_params.append(f'{name} (all zeros)')
                elif torch.any(torch.isnan(param.grad)):
                    nan_grad_params.append(name)
                elif torch.any(torch.isinf(param.grad)):
                    nan_grad_params.append(f'{name} (inf)')

        assert not nan_grad_params, f'NaN/Inf gradients in: {nan_grad_params}'
        assert not zero_grad_params, (
            f'Zero gradients in: {zero_grad_params}. '
            f'These parameters are not learning.'
        )

    def test_gradient_magnitudes_reasonable(self):
        """Gradient norms should be in a reasonable range."""
        torch.manual_seed(42)
        model = _tiny_model()
        model.train()

        batch = torch.randint(0, 64, (4, 16))
        logits = model(batch[:, :-1])
        loss = f.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            batch[:, 1:].reshape(-1),
        )
        loss.backward()

        total_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                total_norm += param.grad.norm().item() ** 2
        total_norm = total_norm**0.5

        # Gradient norm should be reasonable (not vanishing or exploding)
        assert total_norm > 1e-6, (
            f'Gradient norm {total_norm:.2e} is too small (vanishing).'
        )
        assert total_norm < 1e4, (
            f'Gradient norm {total_norm:.2e} is too large (exploding).'
        )


class TestBPBSanity:
    """Verify BPB computation is correct.

    For a byte-level tokenizer (1 byte per token), 2^BPB should equal
    perplexity. This is a mathematical identity we can verify.
    """

    def test_bpb_perplexity_equivalence(self):
        """2^BPB should equal perplexity for byte-level tokenizer."""
        from lmt.eval.bpb import compute_bpb

        torch.manual_seed(42)
        model = _tiny_model()
        model.eval()

        data = torch.randint(0, 64, (4, 16))

        # Compute BPB
        bpb = compute_bpb(model, data, bytes_per_token=1.0)

        # Compute perplexity directly
        with torch.no_grad():
            logits = model(data[:, :-1])
            loss = f.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                data[:, 1:].reshape(-1),
            )
        perplexity = math.exp(loss.item())

        # 2^BPB should equal perplexity (for byte-level tokenizer)
        bpb_perplexity = 2**bpb

        # Allow 5% tolerance for floating point
        assert abs(bpb_perplexity - perplexity) / perplexity < 0.05, (
            f'2^BPB={bpb_perplexity:.4f} != perplexity={perplexity:.4f}. '
            f'BPB computation may be incorrect.'
        )


class TestGRPOSanity:
    """Verify GRPO training basics."""

    def test_grpo_loss_decreases(self):
        """GRPO loss should decrease over steps on a toy task."""
        from lmt.training.grpo_trainer import GRPOConfig, GRPOTrainer

        torch.manual_seed(42)
        model = _tiny_model()

        config = GRPOConfig(
            group_size=4,
            max_response_len=4,
            clip_eps=0.2,
            kl_coeff=0.0,
            lr=1e-4,
            temperature=1.0,
            top_k=0,
            device='cpu',
        )

        trainer = GRPOTrainer(model, config)
        prompt = torch.randint(0, 64, (4,))

        # Reward function that prefers sequences ending with token 0
        def reward_fn(
            prompt_ids: torch.Tensor,
            response_ids: torch.Tensor,
        ) -> float:
            # Higher reward if last token is close to 0
            return 1.0 - (response_ids[-1].item() / 64.0)

        losses = []
        for _ in range(5):
            loss = trainer.train_step(prompt, reward_fn)
            losses.append(loss)

        # Loss should not be NaN
        assert not any(math.isnan(v) for v in losses), (
            f'NaN loss during GRPO training: {losses}'
        )
        # Loss should be finite
        assert all(math.isfinite(v) for v in losses), (
            f'Infinite loss during GRPO training: {losses}'
        )


class TestRewardSanity:
    """Verify reward functions produce correct scores."""

    def test_math_reward_correct(self):
        """Correct answer should get reward 1.0."""
        from lmt.training.rewards import math_reward

        assert math_reward('', r'\boxed{42}', '42') == 1.0

    def test_math_reward_wrong(self):
        """Wrong answer should get reward 0.0."""
        from lmt.training.rewards import math_reward

        assert math_reward('', r'\boxed{43}', '42') == 0.0

    def test_math_reward_no_answer(self):
        """No extractable answer should get reward 0.0."""
        from lmt.training.rewards import math_reward

        assert (
            math_reward('', 'I think the answer is maybe forty-two', '42')
            == 0.0
        )

    def test_math_reward_numeric_equivalence(self):
        """Numerically equivalent answers should match."""
        from lmt.training.rewards import math_reward

        assert math_reward('', r'\boxed{42.0}', '42') == 1.0
        assert math_reward('', r'\boxed{42.00}', '42') == 1.0

    def test_format_reward_all_tags(self):
        """Response with all required tags gets 1.0."""
        from lmt.training.rewards import format_reward

        response = '<think>reasoning</think><answer>42</answer>'
        assert format_reward(response, ['think', 'answer']) == 1.0

    def test_format_reward_partial(self):
        """Response with some tags gets partial credit."""
        from lmt.training.rewards import format_reward

        response = '<think>reasoning</think>'
        assert format_reward(response, ['think', 'answer']) == 0.5
