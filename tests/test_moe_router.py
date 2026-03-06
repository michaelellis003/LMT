"""Unit tests for MoE Top-K Router."""

import torch

from lmt.layers.ffn.moe_router import TopKRouter


class TestTopKRouter:
    """Test suite for TopKRouter."""

    def test_initialization(self):
        """Test router initializes with correct gate."""
        router = TopKRouter(
            num_experts=8, d_model=64, top_k=2
        )
        assert router.gate.in_features == 64
        assert router.gate.out_features == 8
        assert router.top_k == 2

    def test_forward_shapes(self):
        """Test output shapes of gate values and indices."""
        router = TopKRouter(
            num_experts=8, d_model=64, top_k=2
        )
        x = torch.randn(2, 16, 64)  # [batch, seq, d_model]
        gate_values, expert_indices, aux_loss = router(x)

        # Flattened: [batch*seq, top_k]
        assert gate_values.shape == (32, 2)
        assert expert_indices.shape == (32, 2)
        assert aux_loss.ndim == 0  # scalar

    def test_gate_values_sum_to_one(self):
        """Top-k gate values should sum to ~1 (renormalized)."""
        router = TopKRouter(
            num_experts=8, d_model=64, top_k=2
        )
        x = torch.randn(1, 8, 64)
        gate_values, _, _ = router(x)
        sums = gate_values.sum(dim=-1)
        torch.testing.assert_close(
            sums, torch.ones_like(sums), atol=1e-5, rtol=1e-5
        )

    def test_expert_indices_in_range(self):
        """Selected expert indices must be in [0, num_experts)."""
        router = TopKRouter(
            num_experts=8, d_model=64, top_k=2
        )
        x = torch.randn(2, 16, 64)
        _, expert_indices, _ = router(x)
        assert (expert_indices >= 0).all()
        assert (expert_indices < 8).all()

    def test_top_k_indices_unique_per_token(self):
        """Each token's top-k experts should be distinct."""
        router = TopKRouter(
            num_experts=8, d_model=64, top_k=2
        )
        x = torch.randn(1, 16, 64)
        _, expert_indices, _ = router(x)
        for i in range(expert_indices.shape[0]):
            selected = expert_indices[i].tolist()
            assert len(selected) == len(set(selected))

    def test_aux_loss_nonnegative(self):
        """Load balancing loss should be non-negative."""
        router = TopKRouter(
            num_experts=8, d_model=64, top_k=2
        )
        x = torch.randn(2, 16, 64)
        _, _, aux_loss = router(x)
        assert aux_loss >= 0

    def test_gradient_flow(self):
        """Test gradients propagate through the router."""
        router = TopKRouter(
            num_experts=8, d_model=64, top_k=2
        )
        x = torch.randn(2, 8, 64, requires_grad=True)
        gate_values, _, aux_loss = router(x)
        (gate_values.sum() + aux_loss).backward()
        assert router.gate.weight.grad is not None
        assert x.grad is not None

    def test_top_k_one(self):
        """Test with top_k=1 (each token goes to one expert)."""
        router = TopKRouter(
            num_experts=4, d_model=32, top_k=1
        )
        x = torch.randn(1, 8, 32)
        gate_values, expert_indices, _ = router(x)
        assert gate_values.shape == (8, 1)
        assert expert_indices.shape == (8, 1)
        # With top_k=1, gate values should all be 1.0
        torch.testing.assert_close(
            gate_values,
            torch.ones_like(gate_values),
            atol=1e-5, rtol=1e-5,
        )
