"""Unit tests for the Multi-Token Prediction head."""

import torch
import torch.nn as nn

from lmt.models.config import ModelConfig
from lmt.models.llama import LLaMA
from lmt.models.multi_token_prediction import MultiTokenPredictionHead


class TestMultiTokenPredictionHead:
    """Test suite for MultiTokenPredictionHead."""

    def _make_model(self, **kwargs):
        """Create a small LLaMA model (BaseModel subclass) for testing."""
        defaults = dict(
            context_length=32,
            vocab_size=256,
            num_layers=2,
            num_heads=4,
            num_kv_heads=2,
            embed_dim=64,
            dropout=0.0,
        )
        defaults.update(kwargs)
        config = ModelConfig(**defaults)
        return LLaMA(config)

    def test_initialization(self):
        """Test MTP head initializes with valid args."""
        model = self._make_model()
        mtp = MultiTokenPredictionHead(model, num_heads=4)

        assert mtp.num_heads == 4
        assert len(mtp.future_layers) == 3  # n-1 extra heads
        assert mtp.model is model

    def test_initialization_single_head(self):
        """With num_heads=1, MTP is equivalent to standard LM."""
        model = self._make_model()
        mtp = MultiTokenPredictionHead(model, num_heads=1)

        assert mtp.num_heads == 1
        assert len(mtp.future_layers) == 0

    def test_forward_training_shapes(self):
        """Test training forward pass returns correct shapes."""
        model = self._make_model()
        mtp = MultiTokenPredictionHead(model, num_heads=4)

        batch_size = 2
        seq_len = 16
        x = torch.randint(0, 256, (batch_size, seq_len))

        logits_list = mtp(x)

        # Should return num_heads sets of logits
        assert len(logits_list) == 4
        for logits in logits_list:
            assert logits.shape == (batch_size, seq_len, 256)

    def test_forward_single_head(self):
        """With num_heads=1, output is same as base model."""
        model = self._make_model()
        mtp = MultiTokenPredictionHead(model, num_heads=1)
        mtp.eval()

        x = torch.randint(0, 256, (2, 8))
        with torch.no_grad():
            logits_list = mtp(x)
            base_logits = model(x)

        assert len(logits_list) == 1
        torch.testing.assert_close(logits_list[0], base_logits)

    def test_compute_loss_shape(self):
        """Test loss computation returns a scalar."""
        model = self._make_model()
        mtp = MultiTokenPredictionHead(model, num_heads=4)

        batch_size = 2
        seq_len = 16
        x = torch.randint(0, 256, (batch_size, seq_len))
        # Targets include the next num_heads tokens
        targets = torch.randint(0, 256, (batch_size, seq_len))

        logits_list = mtp(x)
        loss = mtp.compute_loss(logits_list, targets)

        assert loss.shape == ()
        assert loss.item() > 0

    def test_gradient_flow(self):
        """Test gradients flow through all heads."""
        model = self._make_model()
        mtp = MultiTokenPredictionHead(model, num_heads=3)

        x = torch.randint(0, 256, (2, 8))
        targets = torch.randint(0, 256, (2, 8))

        logits_list = mtp(x)
        loss = mtp.compute_loss(logits_list, targets)
        loss.backward()

        # Check backbone gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, (
                    f'No gradient for backbone.{name}'
                )

        # Check future layer gradients
        for i, layer in enumerate(mtp.future_layers):
            for name, param in layer.named_parameters():
                if param.requires_grad:
                    assert param.grad is not None, (
                        f'No gradient for future_layer[{i}].{name}'
                    )

    def test_shared_unembedding(self):
        """All heads share the same unembedding matrix."""
        model = self._make_model()
        mtp = MultiTokenPredictionHead(model, num_heads=3)

        # The unembedding weight should be the same object
        # for all heads (shared via the base model's out_head)
        base_weight = model.out_head.weight
        assert mtp.shared_out_head.weight is base_weight

    def test_deterministic_eval(self):
        """Eval mode produces identical outputs."""
        model = self._make_model()
        mtp = MultiTokenPredictionHead(model, num_heads=3)
        mtp.eval()

        x = torch.randint(0, 256, (1, 8))
        with torch.no_grad():
            out1 = mtp(x)
            out2 = mtp(x)

        for l1, l2 in zip(out1, out2, strict=True):
            torch.testing.assert_close(l1, l2)

    def test_different_num_heads(self):
        """Test with various numbers of prediction heads."""
        model = self._make_model()

        for n in [1, 2, 4, 8]:
            mtp = MultiTokenPredictionHead(model, num_heads=n)
            x = torch.randint(0, 256, (1, 8))
            logits_list = mtp(x)
            assert len(logits_list) == n

    def test_loss_weighting(self):
        """Test that head losses can be weighted."""
        model = self._make_model()
        mtp = MultiTokenPredictionHead(model, num_heads=3)

        x = torch.randint(0, 256, (2, 8))
        targets = torch.randint(0, 256, (2, 8))
        logits_list = mtp(x)

        # Equal weights
        loss_equal = mtp.compute_loss(logits_list, targets)

        # Custom weights (first head gets all the weight)
        loss_first = mtp.compute_loss(
            logits_list, targets, head_weights=[1.0, 0.0, 0.0]
        )

        # Losses should differ
        assert not torch.allclose(loss_equal, loss_first)

    def test_future_layer_is_transformer_like(self):
        """Each future layer should contain attention + FFN."""
        model = self._make_model()
        mtp = MultiTokenPredictionHead(model, num_heads=3)

        for layer in mtp.future_layers:
            # Should have norm, attention-like, and FFN-like components
            assert isinstance(layer, nn.Module)
            total_params = sum(p.numel() for p in layer.parameters())
            assert total_params > 0

    def test_parameter_count(self):
        """MTP adds parameters for future layers only."""
        model = self._make_model()
        base_params = sum(p.numel() for p in model.parameters())

        mtp = MultiTokenPredictionHead(model, num_heads=4)
        total_params = sum(p.numel() for p in mtp.parameters())

        # MTP adds 3 future layers worth of params
        added_params = total_params - base_params
        assert added_params > 0

        # Each future layer should have similar param count
        # (they're all the same architecture)
        per_layer = added_params / 3
        assert per_layer > 100  # Sanity check -- not trivially small
