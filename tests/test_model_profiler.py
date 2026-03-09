"""Tests for model profiler utility."""

import torch.nn as nn

from lmt.training.profiler import (
    ModelProfile,
    format_profile,
    profile_model,
)


class TestProfileModel:
    """Test profile_model function."""

    def test_total_params(self):
        """Should count total parameters."""
        model = nn.Linear(10, 5, bias=True)
        profile = profile_model(model)
        # 10*5 + 5 = 55
        assert profile.total_params == 55

    def test_trainable_params(self):
        """Should count trainable parameters separately."""
        model = nn.Linear(10, 5)
        model.weight.requires_grad = False
        profile = profile_model(model)
        assert profile.trainable_params == 5  # only bias
        assert profile.total_params == 55

    def test_param_bytes(self):
        """Should compute memory in bytes."""
        model = nn.Linear(10, 5, bias=False)
        profile = profile_model(model)
        # 50 params * 4 bytes (float32) = 200
        assert profile.param_bytes == 200

    def test_layer_breakdown(self):
        """Should provide per-module parameter counts."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.LayerNorm(20),
            nn.Linear(20, 5),
        )
        profile = profile_model(model)
        assert len(profile.layers) > 0

        # Find the two linear layers
        linear_layers = [
            lyr for lyr in profile.layers if 'Linear' in lyr['type']
        ]
        assert len(linear_layers) == 2

    def test_layer_names(self):
        """Should include module names in breakdown."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5),
        )
        profile = profile_model(model)
        names = [lyr['name'] for lyr in profile.layers]
        assert '0' in names  # Sequential names are indices
        assert '2' in names

    def test_no_params_module(self):
        """Should handle modules with no parameters."""
        model = nn.ReLU()
        profile = profile_model(model)
        assert profile.total_params == 0
        assert profile.param_bytes == 0

    def test_nested_model(self):
        """Should handle nested modules."""
        model = nn.Sequential(
            nn.Sequential(
                nn.Linear(10, 10),
                nn.Linear(10, 10),
            ),
            nn.Linear(10, 5),
        )
        profile = profile_model(model)
        # 10*10+10 + 10*10+10 + 10*5+5 = 275
        assert profile.total_params == 275


class TestModelProfile:
    """Test ModelProfile dataclass."""

    def test_human_readable_params(self):
        """Should format large param counts."""
        profile = ModelProfile(
            total_params=1_500_000,
            trainable_params=1_500_000,
            param_bytes=6_000_000,
            layers=[],
        )
        readable = profile.human_readable_params()
        assert '1.50M' in readable

    def test_human_readable_small(self):
        """Should format small param counts."""
        profile = ModelProfile(
            total_params=500,
            trainable_params=500,
            param_bytes=2000,
            layers=[],
        )
        readable = profile.human_readable_params()
        assert '500' in readable

    def test_human_readable_billions(self):
        """Should format billion-scale param counts."""
        profile = ModelProfile(
            total_params=2_500_000_000,
            trainable_params=2_500_000_000,
            param_bytes=10_000_000_000,
            layers=[],
        )
        readable = profile.human_readable_params()
        assert '2.50B' in readable


class TestFormatProfile:
    """Test format_profile function."""

    def test_returns_string(self):
        """Should return a formatted string."""
        model = nn.Linear(10, 5)
        profile = profile_model(model)
        output = format_profile(profile)
        assert isinstance(output, str)

    def test_includes_total(self):
        """Should include total parameter count."""
        model = nn.Linear(10, 5)
        profile = profile_model(model)
        output = format_profile(profile)
        assert 'total' in output.lower() or '55' in output

    def test_includes_layers(self):
        """Should include layer breakdown."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.Linear(20, 5),
        )
        profile = profile_model(model)
        output = format_profile(profile)
        assert 'Linear' in output
