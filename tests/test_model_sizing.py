"""Tests for model sizing utility."""

from lmt.models.sizing import estimate_params, suggest_config


class TestEstimateParams:
    """Test parameter count estimation."""

    def test_basic_estimate(self):
        """Should estimate parameter count for given dimensions."""
        params = estimate_params(
            embed_dim=256,
            num_layers=6,
            ffn_hidden_dim=512,
            vocab_size=32000,
        )
        assert params > 0

    def test_scales_with_layers(self):
        """Should scale roughly linearly with num_layers."""
        params_2 = estimate_params(
            embed_dim=256, num_layers=2, ffn_hidden_dim=512, vocab_size=100
        )
        params_4 = estimate_params(
            embed_dim=256, num_layers=4, ffn_hidden_dim=512, vocab_size=100
        )
        # 4 layers should be roughly 2x the transformer params of 2 layers
        assert params_4 > params_2

    def test_scales_with_embed_dim(self):
        """Should scale quadratically with embed_dim."""
        params_small = estimate_params(
            embed_dim=128, num_layers=2, ffn_hidden_dim=256, vocab_size=100
        )
        params_large = estimate_params(
            embed_dim=256, num_layers=2, ffn_hidden_dim=512, vocab_size=100
        )
        # Doubling embed_dim (and ffn) should roughly 4x transformer params
        assert params_large > 3 * params_small

    def test_vocab_contribution(self):
        """Should include embedding parameters."""
        params_small_vocab = estimate_params(
            embed_dim=256, num_layers=2, ffn_hidden_dim=512, vocab_size=100
        )
        params_large_vocab = estimate_params(
            embed_dim=256, num_layers=2, ffn_hidden_dim=512, vocab_size=50000
        )
        assert params_large_vocab > params_small_vocab

    def test_tied_weights(self):
        """Should reduce count when tie_weights=True."""
        params_tied = estimate_params(
            embed_dim=256,
            num_layers=2,
            ffn_hidden_dim=512,
            vocab_size=32000,
            tie_weights=True,
        )
        params_untied = estimate_params(
            embed_dim=256,
            num_layers=2,
            ffn_hidden_dim=512,
            vocab_size=32000,
            tie_weights=False,
        )
        assert params_tied < params_untied

    def test_returns_int(self):
        """Should return an integer."""
        params = estimate_params(
            embed_dim=64, num_layers=1, ffn_hidden_dim=128, vocab_size=100
        )
        assert isinstance(params, int)


class TestSuggestConfig:
    """Test config suggestion for target parameter count."""

    def test_suggests_config(self):
        """Should return a config dict near target params."""
        config = suggest_config(target_params=1_000_000, vocab_size=32000)
        assert 'embed_dim' in config
        assert 'num_layers' in config
        assert 'ffn_hidden_dim' in config

    def test_near_target(self):
        """Should be within 50% of target."""
        target = 10_000_000
        config = suggest_config(target_params=target, vocab_size=32000)
        actual = estimate_params(
            embed_dim=config['embed_dim'],
            num_layers=config['num_layers'],
            ffn_hidden_dim=config['ffn_hidden_dim'],
            vocab_size=32000,
            tie_weights=True,
        )
        assert 0.5 * target < actual < 1.5 * target

    def test_small_model(self):
        """Should handle small model targets."""
        config = suggest_config(target_params=100_000, vocab_size=256)
        assert config['embed_dim'] > 0
        assert config['num_layers'] > 0

    def test_large_model(self):
        """Should handle larger model targets."""
        config = suggest_config(target_params=100_000_000, vocab_size=32000)
        assert config['embed_dim'] > 0
        assert config['num_layers'] > 0
