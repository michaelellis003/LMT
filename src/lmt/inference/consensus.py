"""Consensus voting for code generation via output clustering.

Clusters code samples by functional equivalence: samples that produce
identical outputs on probe inputs are grouped together, and the largest
cluster's representative is selected. This is inspired by AlphaCode
(DeepMind, 2022) which clustered millions of candidates by output
equivalence on example test inputs.

The key advantage over best-of-N: consensus voting works without
ground-truth test assertions. You only need probe inputs (which can
be extracted from the problem statement or randomly generated).

Usage::

    from lmt.inference.consensus import cluster_by_output

    codes = [
        'def f(x): return x+1',
        'def f(x): return x+1',
        'def f(x): return 0',
    ]
    probes = ['print(f(3))', 'print(f(0))']
    result = cluster_by_output(codes, probes)
    print(f'Selected: {result.best_code}')
    print(
        f'{result.num_clusters} clusters, '
        f'largest has {result.largest_cluster_size}'
    )
"""

from __future__ import annotations

import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class ConsensusResult:
    """Result of consensus voting.

    Attributes:
        best_code: Representative from the largest cluster.
        cluster_sizes: Sizes of each cluster, sorted descending.
        num_clusters: Number of distinct output clusters.
        largest_cluster_size: Size of the winning cluster.
        all_codes: All input code samples.
    """

    best_code: str
    cluster_sizes: list[int]
    num_clusters: int
    largest_cluster_size: int
    all_codes: list[str]


def cluster_by_output(
    codes: list[str],
    probe_calls: list[str],
    timeout: int = 5,
) -> ConsensusResult:
    """Cluster code samples by output equivalence on probe inputs.

    Runs each code sample with the given probe calls and groups
    samples that produce identical stdout. The largest cluster's
    shortest member is selected as the best candidate.

    Samples that crash or timeout form singleton clusters (they
    never agree with anything).

    Args:
        codes: List of code samples to cluster.
        probe_calls: Print statements to run after the code
            (e.g., ``['print(f(3))', 'print(f(0))']``).
        timeout: Maximum execution time per sample in seconds.

    Returns:
        ConsensusResult with the selected code and cluster info.
    """
    # Compute output fingerprint for each sample
    fingerprints: list[str] = []
    crash_counter = 0

    for code in codes:
        script = code + '\n' + '\n'.join(probe_calls)
        try:
            result = subprocess.run(
                [sys.executable, '-c', script],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if result.returncode == 0:
                fingerprints.append(result.stdout)
            else:
                # Crashed — unique fingerprint
                crash_counter += 1
                fingerprints.append(f'__crash_{crash_counter}__')
        except subprocess.TimeoutExpired:
            crash_counter += 1
            fingerprints.append(f'__timeout_{crash_counter}__')

    # Group by fingerprint
    clusters: dict[str, list[int]] = defaultdict(list)
    for i, fp in enumerate(fingerprints):
        clusters[fp].append(i)

    # Sort clusters by size (descending), then by shortest code
    sorted_clusters = sorted(
        clusters.values(),
        key=lambda idxs: (-len(idxs), min(len(codes[i]) for i in idxs)),
    )

    # Pick representative from largest cluster (shortest code)
    largest = sorted_clusters[0]
    best_idx = min(largest, key=lambda i: len(codes[i]))

    cluster_sizes = [len(c) for c in sorted_clusters]

    return ConsensusResult(
        best_code=codes[best_idx],
        cluster_sizes=cluster_sizes,
        num_clusters=len(sorted_clusters),
        largest_cluster_size=len(largest),
        all_codes=codes,
    )
