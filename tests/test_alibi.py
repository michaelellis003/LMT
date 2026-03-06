"""Tests for ALiBi (Attention with Linear Biases)."""

import torch

from lmt.layers.positional.alibi import ALiBi


class TestALiBiInit:
    """Test ALiBi initialization."""

    def test_creates_with_num_heads(self) -> None:
        """ALiBi initializes with given number of heads."""
        alibi = ALiBi(num_heads=8, max_seq_len=1024)
        assert alibi.num_heads == 8

    def test_slopes_shape(self) -> None:
        """Slopes tensor has shape [num_heads]."""
        alibi = ALiBi(num_heads=8, max_seq_len=512)
        assert alibi.slopes.shape == (8,)

    def test_slopes_are_geometric_sequence(self) -> None:
        """Slopes follow 2^(-8/n * i) for i=1..n."""
        n = 8
        alibi = ALiBi(num_heads=n, max_seq_len=512)
        # Expected: 2^(-8/8 * 1), 2^(-8/8 * 2), ... = 1/2, 1/4, ...
        for i in range(n):
            expected = 2.0 ** (-(8.0 / n) * (i + 1))
            assert abs(alibi.slopes[i].item() - expected) < 1e-6, (
                f'Head {i}: expected {expected}, got {alibi.slopes[i].item()}'
            )

    def test_slopes_decrease(self) -> None:
        """Each head's slope is smaller than the previous."""
        alibi = ALiBi(num_heads=8, max_seq_len=512)
        for i in range(1, 8):
            assert alibi.slopes[i] < alibi.slopes[i - 1]

    def test_slopes_all_positive(self) -> None:
        """All slopes are positive (in the (0, 1) range)."""
        alibi = ALiBi(num_heads=16, max_seq_len=512)
        assert (alibi.slopes > 0).all()
        assert (alibi.slopes < 1).all()

    def test_bias_is_buffer(self) -> None:
        """Bias matrix is registered as a buffer (not a parameter)."""
        alibi = ALiBi(num_heads=4, max_seq_len=256)
        assert 'bias' in dict(alibi.named_buffers())
        # Should NOT be a learnable parameter
        param_names = [n for n, _ in alibi.named_parameters()]
        assert 'bias' not in param_names
        assert 'slopes' not in param_names

    def test_non_power_of_two_heads(self) -> None:
        """Works with non-power-of-2 number of heads."""
        alibi = ALiBi(num_heads=6, max_seq_len=512)
        assert alibi.slopes.shape == (6,)
        assert (alibi.slopes > 0).all()


class TestALiBiBiasShape:
    """Test the bias matrix shape and properties."""

    def test_bias_shape(self) -> None:
        """Bias is [num_heads, max_seq_len, max_seq_len]."""
        alibi = ALiBi(num_heads=4, max_seq_len=128)
        assert alibi.bias.shape == (4, 128, 128)

    def test_bias_diagonal_is_zero(self) -> None:
        """Position attending to itself has zero bias."""
        alibi = ALiBi(num_heads=4, max_seq_len=64)
        for h in range(4):
            diag = torch.diagonal(alibi.bias[h])
            assert torch.allclose(diag, torch.zeros_like(diag))

    def test_bias_is_non_positive(self) -> None:
        """All bias values are <= 0 (penalties, not rewards)."""
        alibi = ALiBi(num_heads=8, max_seq_len=128)
        assert (alibi.bias <= 0).all()

    def test_bias_increases_with_distance(self) -> None:
        """Bias magnitude grows linearly with query-key distance."""
        alibi = ALiBi(num_heads=4, max_seq_len=32)
        # For each head, check row 10: bias should be linear in |i-j|
        for h in range(4):
            row = alibi.bias[h, 10, :]
            # Distance 0 should be 0
            assert row[10].item() == 0.0
            # Distance 1 should be -slope
            expected_d1 = -alibi.slopes[h].item()
            assert abs(row[9].item() - expected_d1) < 1e-6
            # Distance 2 should be -2*slope
            expected_d2 = -2 * alibi.slopes[h].item()
            assert abs(row[8].item() - expected_d2) < 1e-6

    def test_different_heads_different_magnitudes(self) -> None:
        """Each head penalizes distance at a different rate."""
        alibi = ALiBi(num_heads=4, max_seq_len=32)
        # Check bias at distance 5 for each head
        biases_at_d5 = [alibi.bias[h, 10, 5].item() for h in range(4)]
        # Head 0 has the largest slope -> strongest penalty (most negative)
        # Head 3 has the smallest slope -> mildest penalty (least negative)
        for i in range(3):
            assert biases_at_d5[i] < biases_at_d5[i + 1]


class TestALiBiGetBias:
    """Test bias extraction for variable sequence lengths."""

    def test_get_bias_shorter_than_max(self) -> None:
        """Can extract bias for a seq_len shorter than max."""
        alibi = ALiBi(num_heads=4, max_seq_len=128)
        bias = alibi.get_bias(seq_len=32)
        assert bias.shape == (4, 32, 32)

    def test_get_bias_equals_max(self) -> None:
        """get_bias at max_seq_len returns the full buffer."""
        alibi = ALiBi(num_heads=4, max_seq_len=64)
        bias = alibi.get_bias(seq_len=64)
        assert torch.allclose(bias, alibi.bias)

    def test_get_bias_is_submatrix(self) -> None:
        """Shorter bias is the top-left submatrix of the full bias."""
        alibi = ALiBi(num_heads=4, max_seq_len=64)
        full = alibi.bias
        sub = alibi.get_bias(seq_len=16)
        assert torch.allclose(sub, full[:, :16, :16])


class TestALiBiDevice:
    """Test device handling."""

    def test_bias_on_same_device(self) -> None:
        """Bias tensor is on the same device as the module."""
        alibi = ALiBi(num_heads=4, max_seq_len=64)
        assert alibi.bias.device.type == 'cpu'

    def test_no_learnable_parameters(self) -> None:
        """ALiBi has zero learnable parameters."""
        alibi = ALiBi(num_heads=8, max_seq_len=512)
        assert sum(p.numel() for p in alibi.parameters()) == 0


class TestALiBiMath:
    """Verify the mathematical properties from the paper."""

    def test_slopes_formula_power_of_two(self) -> None:
        """For 8 heads: slopes = 2^(-1), 2^(-2), ..., 2^(-8)."""
        alibi = ALiBi(num_heads=8, max_seq_len=64)
        expected = [2 ** (-i) for i in range(1, 9)]
        for i, exp in enumerate(expected):
            assert abs(alibi.slopes[i].item() - exp) < 1e-6

    def test_bias_is_negative_distance_times_slope(self) -> None:
        """bias[h, i, j] = -slope_h * |i - j|."""
        alibi = ALiBi(num_heads=4, max_seq_len=16)
        for h in range(4):
            m = alibi.slopes[h].item()
            for i in range(16):
                for j in range(16):
                    expected = -m * abs(i - j)
                    actual = alibi.bias[h, i, j].item()
                    assert abs(actual - expected) < 1e-5, (
                        f'h={h}, i={i}, j={j}: '
                        f'expected {expected}, got {actual}'
                    )

    def test_recency_bias(self) -> None:
        """Closer tokens get higher (less negative) bias."""
        alibi = ALiBi(num_heads=4, max_seq_len=32)
        # For query at position 15, key at 14 vs key at 10
        for h in range(4):
            assert alibi.bias[h, 15, 14] > alibi.bias[h, 15, 10]
