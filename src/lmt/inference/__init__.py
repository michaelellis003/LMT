"""Inference-time compute scaling for code generation.

Provides strategies that improve code generation quality by spending
more compute at inference time: generating multiple candidates,
filtering by execution, voting by consensus, and self-repairing
failures.

Key strategies:

- **Best-of-N**: Generate N samples, execute all, pick the best.
- **Consensus voting**: Cluster by functional equivalence, pick the
  largest cluster.
"""

from lmt.inference.best_of_n import (
    BestOfNConfig,
    BestOfNResult,
    best_of_n,
    best_of_n_select,
)
from lmt.inference.consensus import (
    ConsensusResult,
    cluster_by_output,
)

__all__ = [
    'BestOfNConfig',
    'BestOfNResult',
    'ConsensusResult',
    'best_of_n',
    'best_of_n_select',
    'cluster_by_output',
]
