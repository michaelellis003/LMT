"""Feed-forward network implementations.

Registry mapping string keys to FFN classes for use in configurable blocks.
"""

from .default import DefaultFeedForward

__all__ = ['DefaultFeedForward']

FFN_REGISTRY: dict[str, type] = {
    'default': DefaultFeedForward,
}
