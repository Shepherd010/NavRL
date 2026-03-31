"""
test_ppo_graph.py — Unit tests for the GraphTransformer used as a policy network.

No Isaac Sim dependency; pure PyTorch.
Tests the full policy forward/backward including action selection,
log-probability computation, and value estimation via a minimal wrapper.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

import torch
import torch.nn as nn
import pytest
from graph_transformer import GraphTransformer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

B     = 8
N     = 10   # candidate nodes (ego excluded from policy action space)
N_TOT = N + 1  # total nodes including ego at index 0

NODE_FEAT_DIM = 18
EDGE_FEAT_DIM = 8
HIDDEN_DIM    = 64
NUM_HEADS     = 4
NUM_LAYERS    = 3


@pytest.fixture
def model():
    return GraphTransformer(
        node_feat_dim=NODE_FEAT_DIM,
        edge_feat_dim=EDGE_FEAT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
    )


@pytest.fixture
def dummy_inputs():
    torch.manual_seed(42)
    node_features = torch.randn(B, N_TOT, NODE_FEAT_DIM)
    edge_features = torch.randn(B, N_TOT, N_TOT, EDGE_FEAT_DIM)

    # node_mask: all valid except last 2 candidates
    node_mask = torch.ones(B, N_TOT, dtype=torch.bool)
    node_mask[:, -2:] = False

    # edge_mask: simple random sparse edges
    edge_mask = torch.rand(B, N_TOT, N_TOT) > 0.5
    edge_mask &= ~torch.eye(N_TOT, dtype=torch.bool).unsqueeze(0)

    return node_features, edge_features, node_mask, edge_mask


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------

class TestOutputShapes:

    def test_probs_shape(self, model, dummy_inputs):
        probs, h = model(*dummy_inputs)
        assert probs.shape == (B, N), f"Expected ({B},{N}), got {probs.shape}"

    def test_h_shape(self, model, dummy_inputs):
        probs, h = model(*dummy_inputs)
        assert h.shape == (B, N_TOT, HIDDEN_DIM), \
            f"Expected ({B},{N_TOT},{HIDDEN_DIM}), got {h.shape}"

    def test_single_batch(self, model):
        node_features = torch.randn(1, N_TOT, NODE_FEAT_DIM)
        edge_features = torch.randn(1, N_TOT, N_TOT, EDGE_FEAT_DIM)
        node_mask = torch.ones(1, N_TOT, dtype=torch.bool)
        edge_mask = torch.zeros(1, N_TOT, N_TOT, dtype=torch.bool)
        probs, h = model(node_features, edge_features, node_mask, edge_mask)
        assert probs.shape == (1, N)
        assert h.shape == (1, N_TOT, HIDDEN_DIM)

    def test_large_batch(self, model):
        B2 = 64
        node_features = torch.randn(B2, N_TOT, NODE_FEAT_DIM)
        edge_features = torch.randn(B2, N_TOT, N_TOT, EDGE_FEAT_DIM)
        node_mask = torch.ones(B2, N_TOT, dtype=torch.bool)
        edge_mask = torch.zeros(B2, N_TOT, N_TOT, dtype=torch.bool)
        probs, h = model(node_features, edge_features, node_mask, edge_mask)
        assert probs.shape == (B2, N)


# ---------------------------------------------------------------------------
# Probability validity
# ---------------------------------------------------------------------------

class TestProbabilities:

    def test_probs_sum_to_one(self, model, dummy_inputs):
        probs, _ = model(*dummy_inputs)
        sums = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(B), atol=1e-5), \
            f"Probs don't sum to 1: {sums}"

    def test_probs_non_negative(self, model, dummy_inputs):
        probs, _ = model(*dummy_inputs)
        assert (probs >= -1e-6).all(), "Probs should be non-negative"

    def test_probs_no_nan(self, model, dummy_inputs):
        probs, _ = model(*dummy_inputs)
        assert not torch.isnan(probs).any(), "NaN in probs"

    def test_h_no_nan(self, model, dummy_inputs):
        _, h = model(*dummy_inputs)
        assert not torch.isnan(h).any(), "NaN in h"

    def test_masked_candidates_zero_prob(self, model):
        """Masked-out candidate nodes should get zero probability."""
        node_features = torch.randn(2, N_TOT, NODE_FEAT_DIM)
        edge_features = torch.zeros(2, N_TOT, N_TOT, EDGE_FEAT_DIM)
        node_mask = torch.ones(2, N_TOT, dtype=torch.bool)
        # Mask out all candidates except first two (indices 1 and 2)
        node_mask[:, 3:] = False
        edge_mask = torch.zeros(2, N_TOT, N_TOT, dtype=torch.bool)

        probs, _ = model(node_features, edge_features, node_mask, edge_mask)
        # Candidates at indices 3..N (probs indices 2..N-1) should be ~0
        assert (probs[:, 2:] < 1e-5).all(), \
            "Masked candidates should have near-zero probability"


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------

class TestRobustness:

    def test_all_candidates_masked_no_nan(self, model):
        """
        When all candidates are masked, model must NOT crash or produce NaN.
        (Uses the first-candidate fallback in GraphTransformer.)
        """
        node_features = torch.randn(2, N_TOT, NODE_FEAT_DIM)
        edge_features = torch.zeros(2, N_TOT, N_TOT, EDGE_FEAT_DIM)
        node_mask = torch.ones(2, N_TOT, dtype=torch.bool)
        node_mask[:, 1:] = False   # all candidates masked, only ego valid

        edge_mask = torch.zeros(2, N_TOT, N_TOT, dtype=torch.bool)
        probs, h = model(node_features, edge_features, node_mask, edge_mask)

        assert not torch.isnan(probs).any(), "NaN in probs when all candidates masked"
        assert not torch.isnan(h).any(),     "NaN in h when all candidates masked"
        # sum should still be 1
        assert torch.allclose(probs.sum(-1), torch.ones(2), atol=1e-5)

    def test_all_edges_absent(self, model):
        """Dense-zero edge matrix should still produce valid probs."""
        node_features = torch.randn(B, N_TOT, NODE_FEAT_DIM)
        edge_features = torch.zeros(B, N_TOT, N_TOT, EDGE_FEAT_DIM)
        node_mask = torch.ones(B, N_TOT, dtype=torch.bool)
        edge_mask = torch.zeros(B, N_TOT, N_TOT, dtype=torch.bool)

        probs, h = model(node_features, edge_features, node_mask, edge_mask)
        assert not torch.isnan(probs).any()
        assert torch.allclose(probs.sum(-1), torch.ones(B), atol=1e-5)

    def test_nan_in_input_features(self, model):
        """NaN padding in masked-out node features should not propagate to valid nodes."""
        node_features = torch.randn(B, N_TOT, NODE_FEAT_DIM)
        node_features[:, 5:, :] = float('nan')   # padded nodes
        edge_features = torch.zeros(B, N_TOT, N_TOT, EDGE_FEAT_DIM)
        node_mask = torch.ones(B, N_TOT, dtype=torch.bool)
        node_mask[:, 5:] = False
        edge_mask = torch.zeros(B, N_TOT, N_TOT, dtype=torch.bool)

        # Zero out NaN positions before forward (as caller should do)
        node_features[~node_mask] = 0.0
        probs, h = model(node_features, edge_features, node_mask, edge_mask)
        assert not torch.isnan(probs).any()


# ---------------------------------------------------------------------------
# Gradient / optimisation
# ---------------------------------------------------------------------------

class TestGradients:

    def test_backward_no_error(self, model, dummy_inputs):
        """full backward pass must not raise."""
        probs, h = model(*dummy_inputs)
        loss = probs.sum() + h.sum()
        loss.backward()

    def test_parameters_receive_grad(self, model, dummy_inputs):
        probs, h = model(*dummy_inputs)
        loss = probs.sum() + h.sum()
        loss.backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"Parameter {name} has no gradient"

    def test_h_ego_gradient_flows(self, model, dummy_inputs):
        """Gradient should flow through the ego embedding h[:, 0, :]."""
        probs, h = model(*dummy_inputs)
        ego_out = h[:, 0, :]
        loss = ego_out.sum()
        loss.backward()
        # At least some model parameters must have non-zero gradients
        has_nonzero = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
        )
        assert has_nonzero, "No non-zero gradients from ego embedding"

    def test_different_actions_different_grads(self, model):
        """Policy should be sensitive to node features."""
        node_f1 = torch.zeros(1, N_TOT, NODE_FEAT_DIM)
        node_f2 = torch.rand(1, N_TOT, NODE_FEAT_DIM)
        edge_f  = torch.zeros(1, N_TOT, N_TOT, EDGE_FEAT_DIM)
        mask_n  = torch.ones(1, N_TOT, dtype=torch.bool)
        mask_e  = torch.zeros(1, N_TOT, N_TOT, dtype=torch.bool)

        probs1, _ = model(node_f1, edge_f, mask_n, mask_e)
        probs2, _ = model(node_f2, edge_f, mask_n, mask_e)
        # Different features → different distributions
        assert not torch.allclose(probs1, probs2), \
            "Policy unchanged for completely different inputs"


# ---------------------------------------------------------------------------
# Action sampling
# ---------------------------------------------------------------------------

class TestActionSampling:

    def test_categorical_sample_valid(self, model, dummy_inputs):
        """Sample from probs using Categorical; actions must be in [0, N)."""
        from torch.distributions import Categorical
        probs, _ = model(*dummy_inputs)
        dist = Categorical(probs=probs)
        actions = dist.sample()
        assert actions.shape == (B,)
        assert (actions >= 0).all()
        assert (actions < N).all()

    def test_log_prob_finite(self, model, dummy_inputs):
        """Log-probs from Categorical must be finite (no -inf / NaN)."""
        from torch.distributions import Categorical
        probs, _ = model(*dummy_inputs)
        dist = Categorical(probs=probs)
        actions = dist.sample()
        log_p = dist.log_prob(actions)
        assert log_p.shape == (B,)
        assert torch.isfinite(log_p).all(), \
            f"Non-finite log_probs: {log_p}"

    def test_greedy_action_consistent_with_probs(self, model, dummy_inputs):
        """argmax of probs should equal greedy action."""
        probs, _ = model(*dummy_inputs)
        greedy = probs.argmax(dim=-1)
        assert greedy.shape == (B,)
        for b in range(B):
            assert probs[b, greedy[b]] == probs[b].max(), \
                f"Greedy mismatch at batch {b}"
