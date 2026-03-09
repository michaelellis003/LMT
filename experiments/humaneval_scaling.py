"""HumanEval inference-time scaling experiment.

Evaluates how inference-time compute scaling improves code generation
on the HumanEval benchmark. Compares:
1. Greedy baseline (pass@1)
2. Best-of-N with execution filtering
3. Consensus voting

Uses Qwen2.5-Coder-0.5B-Instruct as the base model (~494M params).

Usage::

    # Quick pilot (10 problems, small N)
    python experiments/humaneval_scaling.py \
        --n-problems 10 --n-samples 5

    # Full evaluation
    python experiments/humaneval_scaling.py --n-samples 50

Requirements: transformers, datasets (not project dependencies —
install with: uv pip install --python .venv/bin/python transformers datasets)
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError as e:
    raise ImportError(
        'This experiment requires transformers. Install with:\n'
        '  uv pip install --python .venv/bin/python transformers datasets'
    ) from e

from lmt.eval.code_execution import ExecutionResult, execute_code
from lmt.eval.pass_at_k import pass_at_k_problems
from lmt.inference.best_of_n import best_of_n_select
from lmt.inference.consensus import cluster_by_output

# --- Configuration ---

MODEL_NAME = 'Qwen/Qwen2.5-Coder-0.5B-Instruct'
STOP_TOKENS = ['\nclass ', '\ndef ', '\n# ', '\n@', '\nprint(', '\nif __']
MAX_NEW_TOKENS = 256
RESULTS_DIR = Path('experiments/results')


@dataclass
class ProblemResult:
    """Result for a single HumanEval problem."""

    task_id: str
    greedy_correct: bool
    greedy_reward: float
    best_of_n_correct: bool
    best_of_n_reward: float
    consensus_correct: bool
    consensus_reward: float
    n_samples: int
    n_correct_samples: int
    generation_time: float


def load_humaneval(n_problems: int | None = None) -> list[dict]:
    """Load HumanEval problems from HuggingFace datasets."""
    from datasets import load_dataset

    ds = load_dataset('openai/openai_humaneval', split='test')
    problems = list(ds)
    if n_problems is not None:
        problems = problems[:n_problems]
    print(f'Loaded {len(problems)} HumanEval problems')
    return problems


def load_model(
    model_name: str = MODEL_NAME,
) -> tuple:
    """Load model and tokenizer."""
    print(f'Loading {model_name}...')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32,
    )
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  Loaded ({n_params / 1e6:.0f}M params)')
    return model, tokenizer


def generate_completions(
    model,
    tokenizer,
    prompt: str,
    n_samples: int = 1,
    temperature: float = 0.8,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> list[str]:
    """Generate n completions for a HumanEval prompt.

    Uses num_return_sequences for batched generation when
    sampling (n_samples > 1), which is much faster than a
    sequential loop.
    """
    inputs = tokenizer(prompt, return_tensors='pt')
    input_len = inputs['input_ids'].shape[1]

    with torch.no_grad():
        if temperature < 0.01:
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id,
            )
        else:
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.95,
                num_return_sequences=n_samples,
                pad_token_id=tokenizer.eos_token_id,
            )

    completions = []
    for i in range(outputs.shape[0]):
        completion = tokenizer.decode(
            outputs[i][input_len:],
            skip_special_tokens=True,
        )
        # Truncate at stop tokens
        for stop in STOP_TOKENS:
            if stop in completion:
                completion = completion[: completion.index(stop)]
        completions.append(completion)
    return completions


def run_humaneval_tests(
    code: str,
    test_code: str,
    timeout: int = 5,
) -> ExecutionResult:
    """Run HumanEval tests as a single script.

    HumanEval tests are structured as function definitions
    (not simple assertions), so we concatenate code + tests
    and execute as one block.
    """
    full_script = f'{code}\n{test_code}'
    result = execute_code(full_script, timeout=timeout)
    # If no error, all tests passed (HumanEval tests raise on failure)
    passed = not result.timed_out and not result.stderr
    return ExecutionResult(
        stdout=result.stdout,
        stderr=result.stderr,
        passed=1 if passed else 0,
        failed=0 if passed else 1,
        total=1,
        timed_out=result.timed_out,
    )


def evaluate_problem(
    model,
    tokenizer,
    problem: dict,
    n_samples: int,
) -> ProblemResult:
    """Evaluate a single HumanEval problem with all strategies."""
    task_id = problem['task_id']
    prompt = problem['prompt']
    tests = problem['test']
    entry_point = problem['entry_point']

    # Build test string from HumanEval format
    # HumanEval tests are functions — we need to call them
    test_code = f'{tests}\ncheck({entry_point})'

    start = time.time()

    # 1. Greedy baseline
    greedy_completions = generate_completions(
        model, tokenizer, prompt, n_samples=1, temperature=0.0
    )
    greedy_code = prompt + greedy_completions[0]
    greedy_exec = run_humaneval_tests(greedy_code, test_code, timeout=5)

    # 2. Sample N completions
    sample_completions = generate_completions(
        model, tokenizer, prompt, n_samples=n_samples, temperature=0.8
    )

    # Execute all samples
    sample_codes = [prompt + c for c in sample_completions]
    sample_results = [
        run_humaneval_tests(code, test_code, timeout=5)
        for code in sample_codes
    ]

    # 3. Best-of-N selection
    bon_result = best_of_n_select(sample_codes, sample_results)

    # 4. Consensus voting (use entry_point as probe)
    # Build probe calls from the test to extract function calls
    probe_calls = _extract_probe_calls(tests, entry_point)
    if probe_calls:
        consensus_result = cluster_by_output(
            sample_codes, probe_calls, timeout=5
        )
        consensus_code = consensus_result.best_code
        consensus_exec = run_humaneval_tests(
            consensus_code, test_code, timeout=5
        )
    else:
        # Fallback to best-of-N if we can't extract probes
        consensus_code = bon_result.best_code
        consensus_exec = run_humaneval_tests(
            consensus_code, test_code, timeout=5
        )

    elapsed = time.time() - start

    n_correct = sum(1 for r in sample_results if r.all_passed)

    return ProblemResult(
        task_id=task_id,
        greedy_correct=greedy_exec.all_passed,
        greedy_reward=greedy_exec.reward,
        best_of_n_correct=bon_result.best_reward == 1.0,
        best_of_n_reward=bon_result.best_reward,
        consensus_correct=consensus_exec.all_passed,
        consensus_reward=consensus_exec.reward,
        n_samples=n_samples,
        n_correct_samples=n_correct,
        generation_time=elapsed,
    )


def _extract_probe_calls(tests: str, entry_point: str) -> list[str]:
    """Extract simple function calls from HumanEval test code for probing.

    Looks for assert statements and extracts the function call to use
    as probe inputs for consensus voting.
    """
    probes = []
    for line in tests.splitlines():
        line = line.strip()
        if f'{entry_point}(' in line and 'assert' in line:
            # Extract the function call
            start = line.index(f'{entry_point}(')
            # Find matching closing paren
            depth = 0
            end = start
            for i in range(start, len(line)):
                if line[i] == '(':
                    depth += 1
                elif line[i] == ')':
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            call = line[start:end]
            probes.append(f'print({call})')
            if len(probes) >= 3:
                break
    return probes


def main() -> None:
    """Run HumanEval inference-time scaling experiment."""
    parser = argparse.ArgumentParser(
        description='HumanEval inference-time scaling experiment'
    )
    parser.add_argument(
        '--n-problems',
        type=int,
        default=None,
        help='Number of problems to evaluate (default: all 164)',
    )
    parser.add_argument(
        '--n-samples',
        type=int,
        default=10,
        help='Samples per problem for best-of-N (default: 10)',
    )
    parser.add_argument(
        '--model',
        type=str,
        default=MODEL_NAME,
        help=f'Model to evaluate (default: {MODEL_NAME})',
    )
    args = parser.parse_args()

    print('=' * 60)
    print('HumanEval Inference-Time Scaling Experiment')
    print('=' * 60)

    problems = load_humaneval(args.n_problems)
    model, tokenizer = load_model(args.model)

    results: list[ProblemResult] = []
    for i, problem in enumerate(problems):
        print(
            f'\n[{i + 1}/{len(problems)}] {problem["task_id"]}...',
            end=' ',
            flush=True,
        )
        result = evaluate_problem(model, tokenizer, problem, args.n_samples)
        results.append(result)
        status = (
            f'greedy={"Y" if result.greedy_correct else "N"} '
            f'bon={"Y" if result.best_of_n_correct else "N"} '
            f'consensus={"Y" if result.consensus_correct else "N"} '
            f'({result.n_correct_samples}/{result.n_samples} correct) '
            f'[{result.generation_time:.1f}s]'
        )
        print(status)

    # Compute aggregate metrics
    n = len(results)
    greedy_acc = sum(r.greedy_correct for r in results) / n
    bon_acc = sum(r.best_of_n_correct for r in results) / n
    consensus_acc = sum(r.consensus_correct for r in results) / n

    # Compute pass@k for different k values
    pass_at_k_data = [(r.n_samples, r.n_correct_samples) for r in results]
    total_time = sum(r.generation_time for r in results)

    print('\n' + '=' * 60)
    print('RESULTS')
    print('=' * 60)
    print(f'\nModel: {args.model}')
    print(f'Problems: {n}')
    print(f'Samples per problem: {args.n_samples}')
    print(f'Total time: {total_time:.0f}s ({total_time / 60:.1f}min)')
    print(f'\n{"Strategy":<20} {"Accuracy":>10}')
    print('-' * 32)
    print(f'{"Greedy (pass@1)":<20} {greedy_acc:>10.1%}')
    print(f'{"Best-of-N":<20} {bon_acc:>10.1%}')
    print(f'{"Consensus":<20} {consensus_acc:>10.1%}')

    if args.n_samples >= 5:
        print(f'\n{"Metric":<20} {"Value":>10}')
        print('-' * 32)
        for k in [1, 5, 10, 50, 100]:
            if k <= args.n_samples:
                pak = pass_at_k_problems(pass_at_k_data, k=k)
                print(f'{"pass@" + str(k):<20} {pak:>10.1%}')

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_file = RESULTS_DIR / 'humaneval_scaling.json'
    output = {
        'model': args.model,
        'n_problems': n,
        'n_samples': args.n_samples,
        'greedy_accuracy': greedy_acc,
        'best_of_n_accuracy': bon_acc,
        'consensus_accuracy': consensus_acc,
        'total_time_seconds': total_time,
        'per_problem': [
            {
                'task_id': r.task_id,
                'greedy_correct': r.greedy_correct,
                'best_of_n_correct': r.best_of_n_correct,
                'consensus_correct': r.consensus_correct,
                'n_correct_samples': r.n_correct_samples,
                'generation_time': r.generation_time,
            }
            for r in results
        ],
    }
    output_file.write_text(json.dumps(output, indent=2))
    print(f'\nResults saved to {output_file}')


if __name__ == '__main__':
    main()
