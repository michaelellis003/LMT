"""Implementation of various language model architectures."""

from .base import BaseModel
from .config import ModelConfig
from .multi_token_prediction import MultiTokenPredictionHead

__all__ = ['BaseModel', 'ModelConfig', 'MultiTokenPredictionHead']
