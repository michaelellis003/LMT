r"""Code reasoning via GRPO -- Training coding ability with compiler feedback.

Trains a language model to solve coding problems using GRPO with
verifiable rewards from code execution:

- Reward 0.0: Syntax error (code doesn't compile)
- Reward 0.0-1.0: Partial credit based on fraction of tests passing
- Reward 1.0: All test assertions pass

Key insight: Unlike reward models, compiler/test feedback is
perfectly verifiable and deterministic, making it ideal for RLVR.

Usage::

    # Dry run (tiny model, no GPU needed):
    uv run python recipes/reasoning_code.py --dry_run

References:
    DeepSeek-AI (2025). "DeepSeek-R1" (arXiv:2501.12948)
    Oxen.ai (2025). "Training a Rust coding agent with GRPO"
    Posterior-GRPO (arXiv:2508.05170) for code generation
"""

import argparse

import torch

from lmt.models.config import ModelConfig
from lmt.recipes.code_reasoning import CodeProblem, create_code_reward_fn
from lmt.training.grpo_trainer import GRPOConfig, GRPOTrainer
from lmt.training.reproducibility import set_seed


def create_sample_problems() -> list[CodeProblem]:
    """Create sample coding problems for dry-run testing.

    Returns:
        List of simple Python coding problems with tests.
    """
    return [
        CodeProblem(
            prompt=(
                'Write a Python function add(a, b) that returns '
                'the sum of two numbers.'
            ),
            tests=(
                'assert add(1, 2) == 3\n'
                'assert add(0, 0) == 0\n'
                'assert add(-1, 1) == 0'
            ),
        ),
        CodeProblem(
            prompt=(
                'Write a Python function factorial(n) that returns '
                'the factorial of a non-negative integer.'
            ),
            tests=(
                'assert factorial(0) == 1\n'
                'assert factorial(1) == 1\n'
                'assert factorial(5) == 120'
            ),
        ),
        CodeProblem(
            prompt=(
                'Write a Python function is_palindrome(s) that checks '
                'if a string is a palindrome (case-insensitive).'
            ),
            tests=(
                'assert is_palindrome("racecar") == True\n'
                'assert is_palindrome("hello") == False\n'
                'assert is_palindrome("Madam") == True\n'
                'assert is_palindrome("") == True'
            ),
        ),
        CodeProblem(
            prompt=(
                'Write a Python function fibonacci(n) that returns '
                'the nth Fibonacci number (0-indexed).'
            ),
            tests=(
                'assert fibonacci(0) == 0\n'
                'assert fibonacci(1) == 1\n'
                'assert fibonacci(10) == 55'
            ),
        ),
    ]


class MockTokenizer:
    """Simple character-level tokenizer for dry-run testing."""

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
    """Run a minimal GRPO training loop with code reward wiring.

    Validates the full pipeline: coding problems, code extraction,
    execution reward, GRPO training. Uses tiny model on CPU.
    """
    print('=== DRY RUN: Validating GRPO code reasoning pipeline ===')

    set_seed(42)

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

    problems = create_sample_problems()
    print(f'Coding problems loaded: {len(problems)}')

    tokenizer = MockTokenizer(vocab_size=config.vocab_size)

    grpo_config = GRPOConfig(
        group_size=2,
        max_response_len=8,
        clip_eps=0.2,
        kl_coeff=0.0,
        lr=5e-5,
        temperature=1.0,
        device='cpu',
    )

    trainer = GRPOTrainer(model, grpo_config)

    print('Running GRPO training with code rewards...')
    for step, problem in enumerate(problems):
        reward_fn = create_code_reward_fn(problem, tokenizer)
        prompt_ids = torch.tensor(
            tokenizer.encode(problem.prompt), dtype=torch.long
        )
        prompt_ids = prompt_ids[: config.context_length // 2]

        loss = trainer.train_step(prompt_ids, reward_fn)
        print(f'  Step {step}: loss={loss:.4f}')

    print('=== DRY RUN COMPLETE ===')
    print('Code reasoning pipeline validated.')
    print('For real training: use pretrained model + GPU.')


def main() -> None:
    """Entry point for code reasoning recipe."""
    parser = argparse.ArgumentParser(
        description='GRPO Code Reasoning Training',
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
        print(
            'Full code reasoning training not yet implemented.\n'
            'Use --dry_run to validate the pipeline.'
        )


if __name__ == '__main__':
    main()
