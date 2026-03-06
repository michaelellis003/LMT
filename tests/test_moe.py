"""Unit tests for Mixture of Experts FFN layer."""

import torch

from lmt.layers.ffn.moe import MoEFeedForward


class TestMoEFeedForward:
    """Test suite for MoEFeedForward."""

    def test_initialization(self):
        """Test MoE initializes with correct number of experts."""
        moe = MoEFeedForward(d_model=64, num_experts=4, top_k=2)
        assert len(moe.experts) == 4
        assert moe.router.top_k == 2

    def test_forward_shape(self):
        """Test output shape matches input shape."""
        moe = MoEFeedForward(d_model=64, num_experts=4, top_k=2)
        x = torch.randn(2, 8, 64)
        out = moe(x)
        assert out.shape == x.shape

    def test_aux_loss_stored(self):
        """Test that aux_loss is stored as module attribute."""
        moe = MoEFeedForward(d_model=64, num_experts=4, top_k=2)
        x = torch.randn(2, 8, 64)
        moe(x)
        assert hasattr(moe, 'aux_loss')
        assert moe.aux_loss >= 0

    def test_gradient_flow(self):
        """Test gradients flow through all expert parameters."""
        moe = MoEFeedForward(d_model=64, num_experts=4, top_k=2)
        x = torch.randn(2, 8, 64, requires_grad=True)
        out = moe(x)
        (out.sum() + moe.aux_loss).backward()

        # Router should have gradients
        assert moe.router.gate.weight.grad is not None
        assert x.grad is not None

        # At least the selected experts should have gradients
        experts_with_grads = sum(
            1 for e in moe.experts if e.w1.weight.grad is not None
        )
        assert experts_with_grads > 0

    def test_shared_experts(self):
        """Test MoE with shared experts (always active)."""
        moe = MoEFeedForward(
            d_model=64,
            num_experts=4,
            top_k=1,
            num_shared_experts=1,
        )
        assert len(moe.shared_experts) == 1
        x = torch.randn(1, 4, 64)
        out = moe(x)
        assert out.shape == x.shape

    def test_custom_hidden_dim(self):
        """Test experts use specified hidden_dim."""
        moe = MoEFeedForward(d_model=64, hidden_dim=32, num_experts=4, top_k=2)
        assert moe.experts[0].w1.out_features == 32

    def test_deterministic_eval(self):
        """Test eval mode produces identical outputs."""
        moe = MoEFeedForward(d_model=64, num_experts=4, top_k=2)
        moe.eval()
        x = torch.randn(1, 8, 64)
        with torch.no_grad():
            out1 = moe(x)
            out2 = moe(x)
        torch.testing.assert_close(out1, out2)

    def test_edge_case_single_token(self):
        """Test with batch=1, seq_len=1."""
        moe = MoEFeedForward(d_model=64, num_experts=4, top_k=2)
        x = torch.randn(1, 1, 64)
        out = moe(x)
        assert out.shape == x.shape

    def test_many_experts_few_tokens(self):
        """Test when num_experts > num_tokens."""
        moe = MoEFeedForward(d_model=32, num_experts=16, top_k=2)
        x = torch.randn(1, 4, 32)  # Only 4 tokens, 16 experts
        out = moe(x)
        assert out.shape == x.shape

    def test_parameter_count(self):
        """MoE should have N * expert_params + router params."""
        moe = MoEFeedForward(d_model=64, hidden_dim=32, num_experts=4, top_k=2)
        # Each SwiGLU expert: 3 * 64 * 32 = 6144
        # Router: 64 * 4 = 256
        expected_expert_params = 4 * 3 * 64 * 32
        expected_router_params = 64 * 4
        total = sum(p.numel() for p in moe.parameters())
        assert total == expected_expert_params + expected_router_params
