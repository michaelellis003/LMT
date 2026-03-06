"""Unit tests for the GPT model."""

import pytest
import torch
import torch.nn as nn

from lmt.models.config import ModelConfig, ModelConfigPresets
from lmt.models.gpt import GPT


class TestGPT:
    """Test suite for the GPT model."""

    def test_gpt_initialization(self):
        """Test GPT model initializes correctly with given config."""
        config = ModelConfigPresets.small_gpt()
        model = GPT(config)

        # Check if model has expected components
        assert hasattr(model, 'tok_embed')
        assert hasattr(model, 'pos_embed')
        assert model.pos_embed is not None
        assert hasattr(model, 'blocks')
        assert hasattr(model, 'final_norm')
        assert hasattr(model, 'out_head')

        # Check dimensions
        assert model.tok_embed.num_embeddings == config.vocab_size
        assert model.tok_embed.embedding_dim == config.embed_dim
        assert model.pos_embed.num_embeddings == config.context_length
        assert model.pos_embed.embedding_dim == config.embed_dim
        assert len(model.blocks) == config.num_layers

    def test_gpt_forward_pass_shape(self):
        """Test that GPT forward pass produces expected output shape."""
        config = ModelConfigPresets.small_gpt(context_length=8)
        model = GPT(config)

        batch_size = 2
        seq_length = 4
        input_ids = torch.randint(
            0, config.vocab_size, (batch_size, seq_length)
        )

        output = model(input_ids)

        expected_shape = (batch_size, seq_length, config.vocab_size)
        assert output.shape == expected_shape

    def test_gpt_forward_pass_deterministic(self):
        """Test that GPT forward pass is deterministic given same input."""
        config = ModelConfigPresets.small_gpt(context_length=8)
        model = GPT(config)
        model.eval()  # Set to eval mode to ensure deterministic behavior

        batch_size = 2
        seq_length = 4
        input_ids = torch.randint(
            0, config.vocab_size, (batch_size, seq_length)
        )

        with torch.no_grad():
            output1 = model(input_ids)
            output2 = model(input_ids)

        torch.testing.assert_close(output1, output2)

    def test_gpt_with_different_configs(self):
        """Test GPT with different model configurations."""
        configs = [
            ModelConfigPresets.small_gpt(),
            ModelConfigPresets.gpt2_124m(context_length=16),
            ModelConfig(
                context_length=32,
                vocab_size=1000,
                num_layers=4,
                num_heads=4,
                embed_dim=128,
                dropout=0.0,
            ),
        ]

        for config in configs:
            model = GPT(config)
            batch_size = 1
            seq_length = min(8, config.context_length)
            input_ids = torch.randint(
                0, config.vocab_size, (batch_size, seq_length)
            )

            output = model(input_ids)
            expected_shape = (batch_size, seq_length, config.vocab_size)
            assert output.shape == expected_shape

    def test_gpt_parameter_count(self):
        """Test that parameter count is reasonable."""
        config = ModelConfigPresets.small_gpt()
        model = GPT(config)

        total_params = sum(p.numel() for p in model.parameters())

        # Rough estimate for small GPT
        # Should be in the order of millions for small model
        assert 1_000_000 < total_params < 100_000_000

    def test_gpt_gradient_flow(self):
        """Test that gradients flow properly through the model."""
        config = ModelConfigPresets.small_gpt(context_length=8)
        model = GPT(config)

        batch_size = 2
        seq_length = 4
        input_ids = torch.randint(
            0, config.vocab_size, (batch_size, seq_length)
        )
        targets = torch.randint(0, config.vocab_size, (batch_size, seq_length))

        output = model(input_ids)
        loss = nn.CrossEntropyLoss()(
            output.view(-1, config.vocab_size), targets.view(-1)
        )
        loss.backward()

        # Check that gradients are computed for all parameters
        for name, param in model.named_parameters():
            assert param.grad is not None, f'No gradient for {name}'
            assert not torch.isnan(param.grad).any(), (
                f'NaN gradient for {name}'
            )

    @pytest.mark.parametrize('seq_length', [1, 4, 8])
    def test_gpt_variable_sequence_lengths(self, seq_length):
        """Test GPT with different sequence lengths."""
        config = ModelConfigPresets.small_gpt(context_length=16)
        model = GPT(config)

        batch_size = 1
        input_ids = torch.randint(
            0, config.vocab_size, (batch_size, seq_length)
        )

        output = model(input_ids)
        expected_shape = (batch_size, seq_length, config.vocab_size)
        assert output.shape == expected_shape

    def test_gpt_device_compatibility(self):
        """Test that model works on different devices."""
        config = ModelConfigPresets.small_gpt(context_length=8)
        model = GPT(config)

        # Test CPU
        input_ids = torch.randint(0, config.vocab_size, (1, 4))
        output_cpu = model(input_ids)
        assert output_cpu.device.type == 'cpu'

        # Test CUDA if available
        if torch.cuda.is_available():
            model_cuda = model.cuda()
            input_ids_cuda = input_ids.cuda()
            output_cuda = model_cuda(input_ids_cuda)
            assert output_cuda.device.type == 'cuda'
