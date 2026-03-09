"""Tests for weight decay parameter groups utility."""

import torch
import torch.nn as nn

from lmt.training.param_groups import get_param_groups


class TestGetParamGroups:
    """Test get_param_groups utility."""

    def test_separates_decay_no_decay(self):
        """Should split params into decay and no-decay groups."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.LayerNorm(20),
            nn.Linear(20, 5),
        )
        groups = get_param_groups(model, weight_decay=0.1)
        assert len(groups) == 2
        assert groups[0]['weight_decay'] == 0.1
        assert groups[1]['weight_decay'] == 0.0

    def test_biases_no_decay(self):
        """Biases should be in the no-decay group."""
        model = nn.Linear(10, 5, bias=True)
        groups = get_param_groups(model, weight_decay=0.1)

        # no-decay group should have the bias
        no_decay_params = groups[1]['params']
        param_names = {
            name
            for name, p in model.named_parameters()
            if any(p is pp for pp in no_decay_params)
        }
        assert 'bias' in param_names

    def test_norm_weights_no_decay(self):
        """LayerNorm and RMSNorm weights should not be decayed."""
        model = nn.Sequential(
            nn.Linear(10, 10),
            nn.LayerNorm(10),
        )
        groups = get_param_groups(model, weight_decay=0.1)

        no_decay_ids = {id(p) for p in groups[1]['params']}
        for name, param in model.named_parameters():
            if (
                'LayerNorm' in type(model[1]).__name__
                and 'weight' in name
                and '1.weight' in name
            ):
                assert id(param) in no_decay_ids

    def test_all_params_accounted(self):
        """Every parameter should appear in exactly one group."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.LayerNorm(20),
            nn.Linear(20, 5),
        )
        groups = get_param_groups(model, weight_decay=0.1)

        all_group_params = set()
        for g in groups:
            for p in g['params']:
                all_group_params.add(id(p))

        model_params = {id(p) for p in model.parameters()}
        assert all_group_params == model_params

    def test_no_duplicate_params(self):
        """No parameter should appear in both groups."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.LayerNorm(20),
            nn.Linear(20, 5),
        )
        groups = get_param_groups(model, weight_decay=0.1)

        decay_ids = {id(p) for p in groups[0]['params']}
        no_decay_ids = {id(p) for p in groups[1]['params']}
        assert decay_ids.isdisjoint(no_decay_ids)

    def test_works_with_adamw(self):
        """Should produce valid param groups for AdamW."""
        model = nn.Linear(10, 5)
        groups = get_param_groups(model, weight_decay=0.01)
        optimizer = torch.optim.AdamW(groups, lr=1e-3)

        x = torch.randn(2, 10)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()  # Should not raise

    def test_embedding_no_decay(self):
        """Embedding weights should not be decayed."""
        model = nn.Sequential(
            nn.Embedding(100, 32),
            nn.Linear(32, 10),
        )
        groups = get_param_groups(model, weight_decay=0.1)

        no_decay_ids = {id(p) for p in groups[1]['params']}
        emb_param = list(model[0].parameters())[0]
        assert id(emb_param) in no_decay_ids

    def test_custom_lr(self):
        """Should pass through additional kwargs."""
        model = nn.Linear(10, 5)
        groups = get_param_groups(model, weight_decay=0.1, lr=3e-4)
        assert groups[0]['lr'] == 3e-4
        assert groups[1]['lr'] == 3e-4

    def test_1d_tensors_no_decay(self):
        """All 1D params (biases, norms) should not be decayed."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.LayerNorm(20),
            nn.Linear(20, 5),
        )
        groups = get_param_groups(model, weight_decay=0.1)

        no_decay_params = groups[1]['params']
        for p in no_decay_params:
            assert p.dim() <= 1 or any(
                p is emb_p
                for m in model.modules()
                if isinstance(m, nn.Embedding)
                for emb_p in m.parameters()
            )
