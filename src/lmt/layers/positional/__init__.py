"""Positional encoding implementations.

Registry mapping string keys to positional encoding classes for use in
configurable blocks.
"""

__all__: list[str] = []

POSITIONAL_REGISTRY: dict[str, type] = {}
