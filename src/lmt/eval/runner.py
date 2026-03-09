"""Unified evaluation runner for LMT models.

Combines math accuracy, code accuracy, and BPB metrics into
a single evaluation interface. No external dependencies beyond
LMT's own eval modules.

Usage::

    from lmt.eval.runner import EvalConfig, EvalRunner

    runner = EvalRunner(model, tokenizer, EvalConfig())
    results = runner.run_all(
        math_items=math_items,
        code_problems=code_problems,
        bpb_tokens=token_sequences,
        bpb_bytes_per_token=3.5,
    )
    print(results.summary())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from lmt.data.math_data import MathDataItem
from lmt.eval.bpb import compute_bpb
from lmt.eval.code_eval import evaluate_code_accuracy
from lmt.eval.math_eval import evaluate_math_accuracy
from lmt.recipes.code_reasoning import CodeProblem


@dataclass
class EvalConfig:
    """Configuration for the evaluation runner.

    Attributes:
        max_new_tokens: Maximum tokens to generate per problem.
        temperature: Sampling temperature (0.0 = greedy).
        device: Device string for evaluation.
        code_timeout: Timeout in seconds for code execution.
    """

    max_new_tokens: int = 256
    temperature: float = 0.0
    device: str = 'cpu'
    code_timeout: int = 10


@dataclass
class EvalResults:
    """Container for evaluation results.

    Attributes:
        metrics: Dict mapping metric names to values.
        details: Dict mapping eval types to per-item details.
    """

    metrics: dict[str, float] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Produce a human-readable summary of all metrics.

        Returns:
            Formatted string with one metric per line.
        """
        lines = ['Evaluation Results:', '-' * 40]
        for name, value in sorted(self.metrics.items()):
            lines.append(f'  {name}: {value:.4f}')
        return '\n'.join(lines)


class EvalRunner:
    """Unified evaluation runner combining all LMT eval modules.

    Args:
        model: An LMT model (``nn.Module``).
        tokenizer: Tokenizer with ``encode``/``decode`` methods.
        config: Evaluation configuration.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        config: EvalConfig | None = None,
    ) -> None:
        """Initialize the runner.

        Args:
            model: An LMT model (``nn.Module``).
            tokenizer: Tokenizer with ``encode``/``decode``.
            config: Evaluation configuration.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or EvalConfig()

    def run_math(
        self,
        items: list[MathDataItem],
    ) -> EvalResults:
        """Run math accuracy evaluation.

        Args:
            items: List of math problems with ground truth.

        Returns:
            EvalResults with math metrics.
        """
        result = evaluate_math_accuracy(
            model=self.model,
            items=items,
            tokenizer=self.tokenizer,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
        )
        return EvalResults(
            metrics={
                'math_accuracy': result['accuracy'],
                'math_num_correct': float(result['num_correct']),
                'math_num_total': float(result['num_total']),
            },
            details={'math': result.get('details', [])},
        )

    def run_code(
        self,
        problems: list[CodeProblem],
    ) -> EvalResults:
        """Run code accuracy evaluation.

        Args:
            problems: List of coding problems with tests.

        Returns:
            EvalResults with code metrics.
        """
        result = evaluate_code_accuracy(
            model=self.model,
            problems=problems,
            tokenizer=self.tokenizer,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            timeout=self.config.code_timeout,
        )
        return EvalResults(
            metrics={
                'code_accuracy': result['accuracy'],
                'code_avg_reward': result['avg_reward'],
                'code_num_correct': float(result['num_correct']),
                'code_num_total': float(result['num_total']),
            },
            details={'code': result.get('details', [])},
        )

    def run_bpb(
        self,
        token_sequences: torch.Tensor,
        bytes_per_token: float,
    ) -> EvalResults:
        """Run BPB (bits per byte) evaluation.

        Args:
            token_sequences: Token IDs [num_sequences, seq_len].
            bytes_per_token: Average UTF-8 bytes per token.

        Returns:
            EvalResults with BPB metric.
        """
        bpb = compute_bpb(
            model=self.model,
            token_sequences=token_sequences,
            bytes_per_token=bytes_per_token,
        )
        return EvalResults(
            metrics={'bpb': bpb},
        )

    def run_all(
        self,
        math_items: list[MathDataItem] | None = None,
        code_problems: list[CodeProblem] | None = None,
        bpb_tokens: torch.Tensor | None = None,
        bpb_bytes_per_token: float | None = None,
    ) -> EvalResults:
        """Run all requested evaluations and merge results.

        Only runs evaluations for which data is provided.

        Args:
            math_items: Math problems (optional).
            code_problems: Code problems (optional).
            bpb_tokens: Token sequences for BPB (optional).
            bpb_bytes_per_token: Bytes per token for BPB.

        Returns:
            Merged EvalResults with all computed metrics.
        """
        merged = EvalResults()

        if math_items is not None:
            math_results = self.run_math(math_items)
            merged.metrics.update(math_results.metrics)
            merged.details.update(math_results.details)

        if code_problems is not None:
            code_results = self.run_code(code_problems)
            merged.metrics.update(code_results.metrics)
            merged.details.update(code_results.details)

        if bpb_tokens is not None and bpb_bytes_per_token is not None:
            bpb_results = self.run_bpb(bpb_tokens, bpb_bytes_per_token)
            merged.metrics.update(bpb_results.metrics)

        return merged
