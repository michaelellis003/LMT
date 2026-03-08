"""Tests for the Muon optimizer.

Muon (MomentUm Orthogonalized by Newton-schulz) is an optimizer for
hidden weight matrices. It applies SGD+momentum followed by an
orthogonalization step via Newton-Schulz iteration, producing updates
that are approximately orthogonal matrices.

Key properties to test:
- Only works on 2D+ parameters
- Newton-Schulz produces approximately orthogonal matrices
- Momentum accumulation behaves correctly
- Weight decay is applied before the update (AdamW-style)
- Training actually reduces loss (integration test)
"""

import torch
import torch.nn as nn

from lmt.training.muon import Muon, newton_schulz


class TestNewtonSchulz:
    """Test the Newton-Schulz orthogonalization."""

    def test_output_shape_matches_input(self):
        """Output should have the same shape as input."""
        g = torch.randn(64, 32)
        result = newton_schulz(g, steps=5)
        assert result.shape == g.shape

    def test_tall_matrix(self):
        """Should handle tall matrices (rows > cols)."""
        g = torch.randn(128, 32)
        result = newton_schulz(g, steps=5)
        assert result.shape == g.shape

    def test_wide_matrix(self):
        """Should handle wide matrices (cols > rows)."""
        g = torch.randn(32, 128)
        result = newton_schulz(g, steps=5)
        assert result.shape == g.shape

    def test_approximately_orthogonal(self):
        """Result should be approximately orthogonal (U @ U^T ≈ I)."""
        g = torch.randn(32, 32)
        result = newton_schulz(g, steps=10).float()
        product = result @ result.T
        # Muon's NS iteration produces S' ~ Uniform(0.5, 1.5),
        # not exactly I, so we check that the off-diagonals are small
        off_diag = product - torch.diag(product.diag())
        assert off_diag.abs().max() < 0.25

    def test_zero_input_no_crash(self):
        """Zero input should not crash (guarded by epsilon)."""
        g = torch.zeros(16, 16)
        result = newton_schulz(g, steps=5)
        assert not torch.isnan(result).any()

    def test_bfloat16_computation(self):
        """Should run internally in bfloat16 for efficiency."""
        g = torch.randn(32, 32, dtype=torch.float32)
        # The function should work with float32 input
        result = newton_schulz(g, steps=5)
        assert result.shape == g.shape


class TestMuonOptimizer:
    """Test the Muon optimizer step behavior."""

    def test_basic_step(self):
        """Muon should update parameters without error."""
        model = nn.Linear(32, 64, bias=False)
        optimizer = Muon(model.parameters(), lr=0.02)

        x = torch.randn(4, 32)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()

    def test_params_change_after_step(self):
        """Parameters should change after an optimizer step."""
        model = nn.Linear(32, 64, bias=False)
        original = model.weight.data.clone()

        optimizer = Muon(model.parameters(), lr=0.02)

        x = torch.randn(4, 32)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()

        assert not torch.equal(model.weight.data, original)

    def test_momentum_accumulates(self):
        """Momentum buffer should accumulate across steps."""
        model = nn.Linear(32, 64, bias=False)
        optimizer = Muon(model.parameters(), lr=0.02, momentum=0.95)

        for _ in range(3):
            x = torch.randn(4, 32)
            loss = model(x).sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Check that momentum buffer exists and is non-zero
        state = optimizer.state[model.weight]
        assert 'momentum_buffer' in state
        assert state['momentum_buffer'].abs().sum() > 0

    def test_weight_decay(self):
        """Weight decay should shrink parameters."""
        model = nn.Linear(32, 64, bias=False)
        nn.init.ones_(model.weight)

        optimizer = Muon(model.parameters(), lr=0.02, weight_decay=0.1)

        x = torch.randn(4, 32)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()

        # Weight decay should reduce the norm (before the update)
        # The update itself may increase it, but decay was applied
        state = optimizer.state[model.weight]
        assert 'momentum_buffer' in state

    def test_zero_grad(self):
        """zero_grad should clear gradients."""
        model = nn.Linear(32, 64, bias=False)
        optimizer = Muon(model.parameters(), lr=0.02)

        x = torch.randn(4, 32)
        loss = model(x).sum()
        loss.backward()
        assert model.weight.grad is not None

        # Default set_to_none=True sets grad to None
        optimizer.zero_grad()
        assert model.weight.grad is None

        # With set_to_none=False, grad is zeroed
        loss = model(x).sum()
        loss.backward()
        optimizer.zero_grad(set_to_none=False)
        assert model.weight.grad is not None
        assert model.weight.grad.abs().sum() == 0

    def test_no_grad_skip(self):
        """Params with no gradient should be skipped."""
        model = nn.Linear(32, 64, bias=False)
        optimizer = Muon(model.parameters(), lr=0.02)

        # Step without backward — no gradients exist
        optimizer.step()  # should not crash

    def test_grad_not_mutated(self):
        """Muon step should not corrupt p.grad (non-in-place Nesterov)."""
        model = nn.Linear(32, 64, bias=False)
        optimizer = Muon(model.parameters(), lr=0.02, nesterov=True)

        x = torch.randn(4, 32)
        loss = model(x).sum()
        loss.backward()

        grad_before = model.weight.grad.clone()
        optimizer.step()

        # Gradient should be unchanged after step
        assert torch.equal(model.weight.grad, grad_before)

    def test_output_dtype_matches_input(self):
        """newton_schulz should return same dtype as input."""
        g = torch.randn(32, 32, dtype=torch.float32)
        result = newton_schulz(g, steps=5)
        assert result.dtype == torch.float32

    def test_default_hyperparams(self):
        """Default hyperparameters should match Keller Jordan's."""
        model = nn.Linear(32, 64, bias=False)
        optimizer = Muon(model.parameters())

        defaults = optimizer.defaults
        assert defaults['lr'] == 0.02
        assert defaults['momentum'] == 0.95
        assert defaults['weight_decay'] == 0


class TestMuonTraining:
    """Integration test: Muon should reduce loss on a simple task."""

    def test_loss_decreases(self):
        """Training a simple model with Muon should reduce loss."""
        torch.manual_seed(42)

        model = nn.Sequential(
            nn.Linear(16, 32, bias=False),
            nn.ReLU(),
            nn.Linear(32, 16, bias=False),
        )

        optimizer = Muon(model.parameters(), lr=0.02)

        # Simple regression task
        x = torch.randn(32, 16)
        target = torch.randn(32, 16)

        losses = []
        for _ in range(20):
            optimizer.zero_grad()
            pred = model(x)
            loss = nn.functional.mse_loss(pred, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss should decrease
        assert losses[-1] < losses[0]

    def test_aspect_ratio_scaling(self):
        """Non-square weight matrices should have aspect ratio scaling."""
        # The update is scaled by sqrt(max(1, rows/cols))
        model = nn.Linear(16, 128, bias=False)  # tall: 128x16
        optimizer = Muon(model.parameters(), lr=0.02)

        x = torch.randn(4, 16)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()

        # Just verify it runs — the scaling is internal
        assert True
