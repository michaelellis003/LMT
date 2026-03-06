r"""YaRN: Yet another RoPE extensioN method.

Extends context length beyond the training window by modifying RoPE
frequencies per-dimension, as described in *YaRN: Efficient Context
Window Extension of Large Language Models* (Peng et al., 2023).

The key insight: different RoPE frequency dimensions encode position
at different scales. High-frequency dimensions capture local patterns
(nearby tokens), while low-frequency dimensions capture global position.
When extending context:

- **High frequencies** should be **preserved** (local patterns don't change)
- **Low frequencies** should be **interpolated** (stretch to cover the
  extended range)
- **Middle frequencies** get a smooth blend via a ramp function

Additionally, YaRN applies an **attention temperature** correction to
compensate for the entropy change in attention distributions at extended
context lengths:

.. math::

    t = 0.1 \ln(s) + 1

where :math:`s` is the scale factor (extended / original length).
Attention logits should be scaled by :math:`1/t`.

References:
    - `YaRN paper <https://arxiv.org/abs/2309.00071>`_
    - `EleutherAI blog <https://blog.eleuther.ai/yarn/>`_
"""

import math

import torch

from lmt.layers.positional.rope import RoPE


class YaRNRoPE(RoPE):
    r"""YaRN-extended Rotary Positional Encoding.

    Modifies RoPE frequencies to support context lengths beyond
    the original training window. High-frequency dimensions are
    preserved, low-frequency dimensions are interpolated, and
    middle dimensions get a smooth blend.

    Args:
        d_model: Feature dimension (must be even).
        original_max_seq_len: Context length the model was trained on.
        extended_max_seq_len: Target extended context length.
        base: Base for RoPE frequency computation.
        beta_fast: Wavelength threshold for high frequencies
            (no interpolation). Default 32.
        beta_slow: Wavelength threshold for low frequencies
            (full interpolation). Default 1.
    """

    def __init__(
        self,
        d_model: int,
        original_max_seq_len: int = 2048,
        extended_max_seq_len: int = 8192,
        base: float = 10000.0,
        beta_fast: int = 32,
        beta_slow: int = 1,
    ) -> None:
        """Initialize YaRN RoPE.

        Args:
            d_model: Feature dimension (must be even).
            original_max_seq_len: Original training context length.
            extended_max_seq_len: Target extended context length.
            base: RoPE base frequency.
            beta_fast: High-frequency boundary (wavelengths below
                this * original_len are preserved).
            beta_slow: Low-frequency boundary (wavelengths above
                this * original_len are fully interpolated).
        """
        # Custom frequency computation — skip RoPE.__init__
        # and call nn.Module.__init__ directly
        torch.nn.Module.__init__(self)

        assert d_model % 2 == 0, 'd_model must be even for RoPE'

        half_d = d_model // 2
        scale = extended_max_seq_len / original_max_seq_len

        # Standard RoPE frequencies
        freqs = 1.0 / (base ** (torch.arange(0, half_d).float() / half_d))

        # Compute per-dimension interpolation factors using the ramp
        # Wavelength for dim i: lambda_i = 2*pi / freq_i
        wavelengths = 2 * math.pi / freqs

        # Ramp boundaries (in wavelength space)
        low_bound = original_max_seq_len * beta_slow
        high_bound = original_max_seq_len * beta_fast

        # Ramp function: 0 = no interpolation, 1 = full interpolation
        # gamma(i) = (wavelength_i - low) / (high - low), clamped to [0, 1]
        if high_bound > low_bound:
            gamma = (wavelengths - low_bound) / (high_bound - low_bound)
            gamma = gamma.clamp(0.0, 1.0)
        else:
            # Edge case: all dimensions get full interpolation
            gamma = torch.ones(half_d)

        # Interpolation factor per dimension:
        # gamma=0 (high freq, short wavelength) -> keep original freq
        # gamma=1 (low freq, long wavelength) -> divide freq by scale
        # This is inverted from the paper's convention — we blend the
        # *inverse* factor: freq_new = freq_old / lerp(1, scale, gamma)
        interp_factor = 1.0 / ((1 - gamma) + gamma / scale)
        adjusted_freqs = freqs * interp_factor

        # Attention temperature (Eq. 20 in YaRN paper)
        self.attn_temperature = 0.1 * math.log(scale) + 1.0

        # Precompute cos/sin caches with adjusted frequencies
        positions = torch.arange(extended_max_seq_len).float()
        angles = torch.outer(positions, adjusted_freqs)

        self.register_buffer('cos_cache', angles.cos())
        self.register_buffer('sin_cache', angles.sin())
