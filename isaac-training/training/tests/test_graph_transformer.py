"""Tests for GraphTransformer (pure PyTorch, no Isaac)."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

import torch
import pytest
from graph_transformer import GraphTransformer


@pytest.fixture
def model():
    return GraphTransformer(
        node_feat_dim=18,
        edge_feat_dim=8,
        hidden_dim=64,
        num_heads=4,
        num_layers=3,
        dropout=0.0,        # deterministic for tests
        use_spd_bias=True,
    )


def make_dummy_graph(B=4, N=10, node_feat_dim=18, edge_feat_dim=8, device='cpu'):
    """N candidate nodes + 1 ego node = N+1 total."""
    torch.manual_seed(0)
    node_features = torch.randn(B, N + 1, node_feat_dim)
    edge_features = torch.randn(B, N + 1, N + 1, edge_feat_dim)

    # Sparse random edge mask (each node connects to ~3 neighbours)
    edge_mask = torch.rand(B, N + 1, N + 1) > 0.7
    edge_mask = edge_mask | edge_mask.transpose(1, 2)    # symmetrize
    eye = torch.eye(N + 1, dtype=torch.bool).unsqueeze(0).expand(B, -1, -1)
    edge_mask = edge_mask & ~eye                          # no self-loops

    node_mask = torch.ones(B, N + 1, dtype=torch.bool)

    return node_features, edge_features, node_mask, edge_mask


# -------------------------------------------------------------------

class TestGraphTransformerShapes:

    def test_probs_shape(self, model):
        B, N = 4, 10
        nf, ef, nm, em = make_dummy_graph(B=B, N=N)
        probs, h = model(nf, ef, nm, em)
        assert probs.shape == (B, N), f"Expected ({B},{N}), got {probs.shape}"

    def test_h_shape(self, model):
        B, N = 4, 10
        nf, ef, nm, em = make_dummy_graph(B=B, N=N)
        probs, h = model(nf, ef, nm, em)
        assert h.shape == (B, N + 1, 64)

    def test_hidden_dim_mismatch_raises(self):
        with pytest.raises(AssertionError):
            GraphTransformer(hidden_dim=65, num_heads=4)  # 65 % 4 != 0


class TestGraphTransformerProbabilities:

    def test_probs_sum_to_one(self, model):
        """Softmax output must sum to 1 over candidate dimension."""
        B, N = 4, 10
        nf, ef, nm, em = make_dummy_graph(B=B, N=N)
        probs, _ = model(nf, ef, nm, em)
        sums = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(B), atol=1e-5), \
            f"probs.sum != 1, got {sums}"

    def test_probs_non_negative(self, model):
        B, N = 4, 10
        nf, ef, nm, em = make_dummy_graph(B=B, N=N)
        probs, _ = model(nf, ef, nm, em)
        assert (probs >= 0).all(), "probs contains negative values"

    def test_probs_no_nan(self, model):
        B, N = 4, 10
        nf, ef, nm, em = make_dummy_graph(B=B, N=N)
        probs, _ = model(nf, ef, nm, em)
        assert not torch.isnan(probs).any(), "NaN in probs"

    def test_h_no_nan(self, model):
        B, N = 4, 10
        nf, ef, nm, em = make_dummy_graph(B=B, N=N)
        _, h = model(nf, ef, nm, em)
        assert not torch.isnan(h).any(), "NaN in h"


class TestGraphTransformerMaskRobustness:

    def test_all_candidate_nodes_masked_fallback(self, model):
        """
        When all candidate nodes are masked out, the model should fallback to
        ensuring at least one candidate is valid — and probs must not be NaN.
        pit #7: using -1e9 instead of float('-inf') prevents NaN in softmax.
        """
        B, N = 2, 10
        nf, ef, nm, em = make_dummy_graph(B=B, N=N)
        # Mask out all candidates (index 1..N); keep ego (index 0) valid
        nm[:, 1:] = False
        probs, h = model(nf, ef, nm, em)
        assert not torch.isnan(probs).any(), \
            "NaN probs with all-masked candidates (check -1e9 vs float('-inf'))"

    def test_partial_mask(self, model):
        """Only some candidates valid — probs should still sum to 1."""
        B, N = 4, 10
        nf, ef, nm, em = make_dummy_graph(B=B, N=N)
        nm[:, 6:] = False  # keep only ego + 5 candidates
        probs, _ = model(nf, ef, nm, em)
        sums = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(B), atol=1e-5)

    def test_sparse_edge_mask(self, model):
        """Very sparse edges should not cause NaN."""
        B, N = 2, 10
        nf, ef, nm, em = make_dummy_graph(B=B, N=N)
        em[:] = False  # no edges at all
        probs, _ = model(nf, ef, nm, em)
        assert not torch.isnan(probs).any()


class TestGraphTransformerGradients:

    def test_backward_pass(self, model):
        """Gradients should flow without error."""
        B, N = 2, 5
        nf, ef, nm, em = make_dummy_graph(B=B, N=N)
        probs, h = model(nf, ef, nm, em)
        loss = probs.sum()
        loss.backward()   # should not raise
        for name, p in model.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"

    def test_h_ego_gradient(self, model):
        """h[:, 0, :] (ego embedding) should have gradient."""
        B, N = 2, 5
        nf, ef, nm, em = make_dummy_graph(B=B, N=N)
        nf.requires_grad_(True)
        probs, h = model(nf, ef, nm, em)
        loss = h[:, 0, :].sum()   # grad through ego embedding
        loss.backward()
        assert nf.grad is not None


class TestSPDMatrix:

    def test_spd_diagonal_zero(self):
        B, N = 2, 6
        em = torch.rand(B, N, N) > 0.5
        em = em | em.transpose(1, 2)
        em[:, torch.arange(N), torch.arange(N)] = False
        spd = GraphTransformer._compute_spd_matrix(em)
        diag = spd[:, torch.arange(N), torch.arange(N)]
        assert (diag == 0).all()

    def test_spd_no_nan(self):
        B, N = 2, 8
        em = torch.zeros(B, N, N, dtype=torch.bool)  # disconnected
        spd = GraphTransformer._compute_spd_matrix(em)
        assert not torch.isnan(spd).any()

    def test_spd_connected_path(self):
        """In a chain 0-1-2, SPD(0,2) should be 2."""
        B, N = 1, 3
        em = torch.zeros(B, N, N, dtype=torch.bool)
        em[0, 0, 1] = em[0, 1, 0] = True
        em[0, 1, 2] = em[0, 2, 1] = True
        spd = GraphTransformer._compute_spd_matrix(em)
        assert spd[0, 0, 2] == 2.0, f"Expected 2.0, got {spd[0,0,2]}"
        assert spd[0, 0, 1] == 1.0
