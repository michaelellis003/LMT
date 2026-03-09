"""Multi-task GRPO training utilities.

Provides iterators for training on multiple reward types (e.g., math
and code) within the same GRPO training loop. Supports alternating
and sequential scheduling strategies.

Usage::

    from lmt.recipes.multi_task import create_multi_task_iterator

    tasks = create_multi_task_iterator(
        task_reward_fns={
            'math': [(prompt1, reward_fn1), (prompt2, reward_fn2)],
            'code': [(prompt3, reward_fn3)],
        },
        strategy='alternate',
    )

    for task_name, prompt, reward_fn in tasks:
        loss = trainer.train_step(prompt, reward_fn)
        print(f'{task_name}: loss={loss:.4f}')
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import torch

# Type alias for reward functions and task collections
RewardFn = Callable[[torch.Tensor, torch.Tensor], float]
TaskItem = tuple[torch.Tensor, RewardFn]
TaskDict = dict[str, list[TaskItem]]
TaskYield = tuple[str, torch.Tensor, RewardFn]


def create_multi_task_iterator(
    task_reward_fns: TaskDict,
    strategy: str = 'alternate',
) -> Iterator[TaskYield]:
    """Create an iterator that yields tasks from multiple reward types.

    Args:
        task_reward_fns: Dict mapping task names to lists of
            (prompt_ids, reward_fn) tuples.
        strategy: Scheduling strategy:
            - ``'alternate'``: Round-robin across tasks.
            - ``'sequential'``: All tasks of one type, then the next.

    Yields:
        Tuples of (task_name, prompt_ids, reward_fn).

    Raises:
        ValueError: If strategy is not recognized.
    """
    if strategy == 'alternate':
        yield from _alternate(task_reward_fns)
    elif strategy == 'sequential':
        yield from _sequential(task_reward_fns)
    else:
        raise ValueError(
            f'Unknown strategy {strategy!r}. '
            f"Supported: 'alternate', 'sequential'"
        )


def _alternate(
    task_reward_fns: TaskDict,
) -> Iterator[TaskYield]:
    """Round-robin across task types."""
    # Create iterators for each task
    iters = {name: iter(items) for name, items in task_reward_fns.items()}
    task_names = list(iters.keys())
    exhausted: set[str] = set()

    while len(exhausted) < len(task_names):
        for name in task_names:
            if name in exhausted:
                continue
            try:
                prompt, reward_fn = next(iters[name])
                yield name, prompt, reward_fn
            except StopIteration:
                exhausted.add(name)


def _sequential(
    task_reward_fns: TaskDict,
) -> Iterator[TaskYield]:
    """All items from each task type in order."""
    for name, items in task_reward_fns.items():
        for prompt, reward_fn in items:
            yield name, prompt, reward_fn
