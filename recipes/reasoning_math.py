r"""Math reasoning via GRPO -- Raschka's recipe adapted for LMT.

Replicates the core finding from Raschka's "reasoning-from-scratch":
GRPO on a pretrained base model (no SFT step) with binary math reward
can significantly improve math reasoning ability.

Key parameters (from Raschka's validated config):
- Model: Qwen3-0.6B (or any pretrained LMT model)
- Group size: G=8
- No KL penalty (kl_coeff=0.0)
- lr=5e-6, 50 training steps
- Binary reward: 1.0 if \boxed{answer} matches ground truth, else 0.0

Usage::

    # From project root, with GPU available:
    uv run python recipes/reasoning_math.py \
        --model_path Qwen/Qwen3-0.6B \
        --output_dir runs/math_reasoning \
        --train_steps 50

    # Dry run (tiny model, no GPU needed):
    uv run python recipes/reasoning_math.py --dry_run

References:
    Raschka, S. (2025). "Build A Reasoning Model (From Scratch)"
    (Manning; GitHub: rasbt/reasoning-from-scratch)
    DeepSeek-AI (2025). "DeepSeek-R1: Incentivizing Reasoning
    Capability in LLMs via Reinforcement Learning" (arXiv:2501.12948)
"""

import argparse
import collections.abc
import json
from pathlib import Path

import torch

from lmt.data.math_data import MathDataItem, format_math_prompt
from lmt.models.config import ModelConfig
from lmt.training.grpo_trainer import GRPOConfig, GRPOTrainer
from lmt.training.reproducibility import ExperimentConfig, set_seed


def create_math_prompts() -> list[MathDataItem]:
    """Create a small set of math problems for testing.

    In a real experiment, this would load GSM8K or MATH from
    HuggingFace datasets. For now, we use a curated set that
    exercises basic arithmetic and algebra.

    Returns:
        List of MathDataItem with prompt and ground_truth.
    """
    problems = [
        ('What is 15 + 27?', '42'),
        ('What is 144 / 12?', '12'),
        ('What is 7 * 8?', '56'),
        ('If x + 5 = 12, what is x?', '7'),
        ('What is 100 - 37?', '63'),
        ('What is 2^8?', '256'),
        ('What is the square root of 49?', '7'),
        ('What is 3/4 + 1/4?', '1'),
    ]

    return [
        MathDataItem(
            prompt=format_math_prompt(problem),
            ground_truth=answer,
        )
        for problem, answer in problems
    ]


def make_reward_fn(
    ground_truth: str,
) -> collections.abc.Callable[[torch.Tensor, torch.Tensor], float]:
    r"""Create a placeholder reward function for a math problem.

    .. warning::

        This is a **stub** that returns random binary rewards.
        It does NOT decode tokens or check against ``ground_truth``.
        A real implementation would use a tokenizer to decode
        ``response_ids`` back to text, then call
        :func:`lmt.training.rewards.math_reward` to compare
        the extracted answer against ``ground_truth``.

    Args:
        ground_truth: The correct answer string (unused in this stub).

    Returns:
        Callable(prompt_ids, response_ids) -> float that returns
        random 0.0 or 1.0 (placeholder).
    """

    def reward_fn(
        prompt_ids: torch.Tensor,
        response_ids: torch.Tensor,
    ) -> float:
        """Return random binary reward (placeholder)."""
        # TODO: Integrate tokenizer for proper decoding
        # Real version: decode response_ids → text → math_reward()
        return float(torch.rand(1).item() > 0.5)

    return reward_fn


def run_dry_run() -> None:
    """Run a minimal GRPO training loop on a tiny model.

    This validates the entire pipeline works without GPU or
    pretrained weights. Uses random data and rewards.
    """
    print('=== DRY RUN: Validating GRPO pipeline ===')

    set_seed(42)

    # Tiny model config
    config = ModelConfig(
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

    from lmt.models.qwen3.qwen3 import Qwen3

    model = Qwen3(config)
    print(f'Model params: {sum(p.numel() for p in model.parameters()):,}')

    grpo_config = GRPOConfig(
        group_size=4,
        max_response_len=8,
        clip_eps=0.2,
        kl_coeff=0.0,
        lr=5e-5,
        temperature=0.7,
        top_k=20,
        device='cpu',
    )

    trainer = GRPOTrainer(model, grpo_config)

    # Create a simple prompt (random tokens)
    prompt = torch.randint(0, 256, (4,))

    # Dummy reward function
    def dummy_reward(
        prompt_ids: torch.Tensor,
        response_ids: torch.Tensor,
    ) -> float:
        return float(torch.rand(1).item())

    print('Running 3 GRPO steps...')
    for step in range(3):
        loss = trainer.train_step(prompt, dummy_reward)
        print(f'  Step {step}: loss={loss:.4f}')

    print('=== DRY RUN COMPLETE ===')
    print('Pipeline validated. Ready for real training with GPU + data.')


def run_training(args: argparse.Namespace) -> None:
    """Set up experiment config for GRPO math reasoning training.

    .. note::

        This is a **placeholder** that saves the experiment config
        and prints the planned training parameters. Actual training
        requires GPU access and pretrained model weights (loaded via
        HuggingFace). Use ``--dry_run`` to validate the pipeline on CPU.

    Args:
        args: Parsed command-line arguments.
    """
    set_seed(args.seed)

    # Log experiment config
    exp_config = ExperimentConfig(
        seed=args.seed,
        model_type='qwen3',
        description=f'GRPO math reasoning, {args.train_steps} steps',
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    exp_config.save(str(output_dir / 'experiment_config.json'))

    print(f'Experiment config saved to {output_dir}/experiment_config.json')
    print(f'Config: {json.dumps(exp_config.to_dict(), indent=2)}')

    # TODO: Load pretrained model from HuggingFace
    # from lmt.models.hf_loader import config_from_hf, load_hf_state_dict
    # For now, this is a placeholder for when GPU + model access is available
    print(
        f'\nWould load model from: {args.model_path}'
        f'\nWould train for {args.train_steps} steps'
        f'\nWould save to: {args.output_dir}'
    )
    print('\nFull training requires GPU + pretrained model weights.')
    print('Use --dry_run to validate the pipeline on CPU.')


def main() -> None:
    """Entry point for math reasoning recipe."""
    parser = argparse.ArgumentParser(
        description='GRPO Math Reasoning Training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '--model_path',
        type=str,
        default='Qwen/Qwen3-0.6B',
        help='HuggingFace model path or local directory',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='runs/math_reasoning',
        help='Output directory for checkpoints and logs',
    )
    parser.add_argument(
        '--train_steps',
        type=int,
        default=50,
        help='Number of GRPO training steps',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility',
    )
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='Run minimal pipeline validation on CPU',
    )

    args = parser.parse_args()

    if args.dry_run:
        run_dry_run()
    else:
        run_training(args)


if __name__ == '__main__':
    main()
