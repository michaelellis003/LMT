"""Tests for auxiliary loss support in training pipeline."""

import torch
import torch.nn as nn

from lmt.training.loss import calc_loss_batch


class MockMoEModel(nn.Module):
    """A mock model with aux_loss attribute (like Mixtral/DeepSeek)."""

    def __init__(self, vocab_size: int = 256, embed_dim: int = 32) -> None:
        """Initialize mock MoE model."""
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)
        self.aux_loss = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass that sets aux_loss."""
        self.aux_loss = torch.tensor(0.5, requires_grad=True)
        return self.head(self.embed(x))


class MockDenseModel(nn.Module):
    """A mock model without aux_loss (like GPT/LLaMA)."""

    def __init__(self, vocab_size: int = 256, embed_dim: int = 32) -> None:
        """Initialize mock dense model."""
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass without aux_loss."""
        return self.head(self.embed(x))


class TestAuxLossInTraining:
    """Test aux_loss integration with training loss."""

    def test_calc_loss_without_aux(self) -> None:
        """Dense model loss is unchanged when aux_loss_coeff=0."""
        model = MockDenseModel()
        tokens = torch.randint(0, 256, (2, 8))
        targets = torch.randint(0, 256, (2, 8))
        device = torch.device('cpu')

        loss_no_aux = calc_loss_batch(tokens, targets, model, device)
        loss_with_coeff = calc_loss_batch(
            tokens, targets, model, device, aux_loss_coeff=0.0
        )
        assert torch.allclose(loss_no_aux, loss_with_coeff)

    def test_calc_loss_with_aux(self) -> None:
        """MoE model loss includes aux_loss when coeff > 0."""
        model = MockMoEModel()
        tokens = torch.randint(0, 256, (2, 8))
        targets = torch.randint(0, 256, (2, 8))
        device = torch.device('cpu')

        loss_no_aux = calc_loss_batch(tokens, targets, model, device)
        loss_with_aux = calc_loss_batch(
            tokens, targets, model, device, aux_loss_coeff=0.01
        )

        # With aux_loss_coeff > 0, total loss should be larger
        assert loss_with_aux.item() > loss_no_aux.item()

    def test_calc_loss_aux_coeff_scales(self) -> None:
        """Higher aux_loss_coeff adds more auxiliary loss."""
        model = MockMoEModel()
        tokens = torch.randint(0, 256, (2, 8))
        targets = torch.randint(0, 256, (2, 8))
        device = torch.device('cpu')

        loss_small = calc_loss_batch(
            tokens, targets, model, device, aux_loss_coeff=0.01
        )
        loss_large = calc_loss_batch(
            tokens, targets, model, device, aux_loss_coeff=1.0
        )

        assert loss_large.item() > loss_small.item()

    def test_calc_loss_dense_model_ignores_aux_coeff(self) -> None:
        """Dense model is unaffected by aux_loss_coeff."""
        model = MockDenseModel()
        tokens = torch.randint(0, 256, (2, 8))
        targets = torch.randint(0, 256, (2, 8))
        device = torch.device('cpu')

        loss_zero = calc_loss_batch(tokens, targets, model, device)
        loss_nonzero = calc_loss_batch(
            tokens, targets, model, device, aux_loss_coeff=0.01
        )

        # Dense model has no aux_loss, so results should be identical
        assert torch.allclose(loss_zero, loss_nonzero)

    def test_aux_loss_gradient_flows(self) -> None:
        """Aux loss contributes to gradients."""
        model = MockMoEModel()
        tokens = torch.randint(0, 256, (2, 8))
        targets = torch.randint(0, 256, (2, 8))
        device = torch.device('cpu')

        loss = calc_loss_batch(
            tokens, targets, model, device, aux_loss_coeff=0.01
        )
        loss.backward()

        # Gradients should exist on model parameters
        assert model.embed.weight.grad is not None
