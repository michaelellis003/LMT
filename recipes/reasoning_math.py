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
import json
from pathlib import Path

import torch

from lmt.data.math_data import load_math_items
from lmt.models.config import ModelConfig
from lmt.recipes.reasoning import create_math_reward_fn
from lmt.training.grpo_trainer import GRPOConfig, GRPOTrainer
from lmt.training.reproducibility import ExperimentConfig, set_seed


def create_mock_gsm8k_data() -> list[dict[str, str]]:
    """Create mock GSM8K-format data for dry runs.

    Returns:
        List of dicts with 'question' and 'answer' fields.
    """
    return [
        {
            'question': 'What is 15 + 27?',
            'answer': 'Fifteen plus twenty-seven is forty-two. #### 42',
        },
        {
            'question': 'What is 144 / 12?',
            'answer': '144 divided by 12 is 12. #### 12',
        },
        {
            'question': 'What is 7 * 8?',
            'answer': 'Seven times eight is fifty-six. #### 56',
        },
        {
            'question': 'If x + 5 = 12, what is x?',
            'answer': 'x = 12 - 5 = 7. #### 7',
        },
    ]


class MockTokenizer:
    """Simple character-level tokenizer for dry-run testing.

    Not suitable for real training but validates the pipeline
    end-to-end including reward function decoding.
    """

    def __init__(self, vocab_size: int = 256) -> None:
        """Initialize mock tokenizer.

        Args:
            vocab_size: Size of the vocabulary.
        """
        self.vocab_size = vocab_size

    def encode(self, text: str) -> list[int]:
        """Encode text to character-level token IDs.

        Args:
            text: Input text string.

        Returns:
            List of integer token IDs.
        """
        return [ord(c) % self.vocab_size for c in text]

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs back to text.

        Args:
            ids: List of token IDs.

        Returns:
            Decoded text string.
        """
        return ''.join(chr(i) for i in ids if 32 <= i < 127)


def run_dry_run() -> None:
    """Run a minimal GRPO training loop with real reward wiring.

    Validates the full pipeline: math data loading, reward bridge,
    GRPO training. Uses mock data and tiny model (no GPU needed).
    """
    print('=== DRY RUN: Validating GRPO math reasoning pipeline ===')

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

    # Load mock math data using the real loader
    mock_rows = create_mock_gsm8k_data()
    items = load_math_items(
        mock_rows, dataset_format='gsm8k', format_prompts=True
    )
    print(f'Math problems loaded: {len(items)}')

    # Create tokenizer for reward bridge
    tokenizer = MockTokenizer(vocab_size=config.vocab_size)

    grpo_config = GRPOConfig(
        group_size=2,
        max_response_len=8,
        clip_eps=0.2,
        kl_coeff=0.0,  # No KL penalty (Raschka's finding)
        lr=5e-5,
        temperature=1.0,
        device='cpu',
    )

    trainer = GRPOTrainer(model, grpo_config)

    # Train on each math problem using real reward bridge
    print('Running GRPO training...')
    for step, item in enumerate(items):
        reward_fn = create_math_reward_fn(item, tokenizer)
        prompt_ids = torch.tensor(
            tokenizer.encode(item.prompt), dtype=torch.long
        )
        # Truncate to fit context
        prompt_ids = prompt_ids[: config.context_length // 2]

        loss = trainer.train_step(prompt_ids, reward_fn)
        print(f'  Step {step}: loss={loss:.4f} | {item.ground_truth}')

    print('=== DRY RUN COMPLETE ===')
    print('Pipeline validated with real reward wiring.')
    print('For real training: use pretrained model + GSM8K + GPU.')


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
