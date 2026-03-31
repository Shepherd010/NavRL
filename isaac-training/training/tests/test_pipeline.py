"""
test_pipeline.py — End-to-end integration test: TopoExtractor → GraphTransformer → SafetyShieldQP

No Isaac Sim dependency; pure PyTorch + NumPy + OSQP.

Tests the full inference pipeline that runs inside the env step loop.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

import time
import torch
import numpy as np
import pytest

from topo_extractor import TopoExtractor
from graph_transformer import GraphTransformer
from safety_shield import SafetyShieldQP


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NUM_RAYS  = 144
N_CAND    = 10       # max candidate nodes
N_TOT     = N_CAND + 1  # +1 ego

NODE_FEAT_DIM = 18
EDGE_FEAT_DIM = 8


def make_topo_cfg():
    class Cfg:
        max_nodes    = N_CAND
        node_feat_dim = NODE_FEAT_DIM
        edge_feat_dim = EDGE_FEAT_DIM
        lidar_range  = 10.0
        grid_size    = 50    # smaller for speed
        safe_radius  = 0.4
        grad_threshold = 0.3
        nms_radius   = 0.5
        k_neighbors  = 3
        max_edge_length = 5.0
        edge_dist_threshold = 5.0
    return Cfg()


def make_gt_cfg():
    class Cfg:
        hidden_dim = 64
        num_heads  = 4
        num_layers = 2
        dropout    = 0.0
    return Cfg()


def make_qp_cfg():
    class Cfg:
        safe_distance  = 0.8
        cbf_alpha      = 2.0
        v_max          = 3.0
        qp_relaxation_weight = 1000.0
        solver_timeout = 0.01
    return Cfg()


def make_dummy_lidar_inputs(B: int, device: str = 'cpu'):
    """
    Simulates B drones surrounded by random obstacles.
    ray_hits_w : (B, NUM_RAYS, 3) — each ray hits a point in world frame
    ray_pos_w  : (B, 3)           — ego positions
    dyn_obs    : (B, 4, 10)       — dynamic obstacles (already squeezed)
    ego_vel    : (B, 3)
    target_pos : (B, 3)
    """
    torch.manual_seed(0)
    ego_pos = torch.rand(B, 3) * 10.0

    angles = torch.linspace(0, 2 * torch.pi, NUM_RAYS)
    dirs   = torch.stack([torch.cos(angles), torch.sin(angles), torch.zeros(NUM_RAYS)], dim=-1)
    ranges = (torch.rand(B, NUM_RAYS) * 8.0 + 0.5).unsqueeze(-1)  # 0.5..8.5 m
    ray_hits = ego_pos.unsqueeze(1) + ranges * dirs.unsqueeze(0)   # (B, 144, 3)

    dyn_obs = torch.zeros(B, 4, 10)   # 4 dynamic obstacles, already squeezed
    dyn_obs[:, :, :3] = torch.rand(B, 4, 3) * 5.0   # positions
    dyn_obs[:, :, 6]  = 0.3                           # radius in col 6

    ego_vel    = torch.rand(B, 3) * 1.0
    target_pos = torch.rand(B, 3) * 10.0 + 10.0   # far away

    return (
        ray_hits.to(device),
        ego_pos.to(device),
        dyn_obs.to(device),
        ego_vel.to(device),
        target_pos.to(device),
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def extractor():
    return TopoExtractor(make_topo_cfg())


@pytest.fixture(scope='module')
def policy():
    cfg = make_gt_cfg()
    return GraphTransformer(
        node_feat_dim=NODE_FEAT_DIM,
        edge_feat_dim=EDGE_FEAT_DIM,
        hidden_dim=cfg.hidden_dim,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
    )


@pytest.fixture(scope='module')
def shield():
    cfg = make_qp_cfg()
    return SafetyShieldQP(
        relaxation_weight=cfg.qp_relaxation_weight,
        v_max=cfg.v_max,
        safe_distance=cfg.safe_distance,
        cbf_alpha=cfg.cbf_alpha,
    )


# ---------------------------------------------------------------------------
# Stage 1: TopoExtractor
# ---------------------------------------------------------------------------

class TestTopoExtractorStage:

    def test_output_keys(self, extractor):
        B = 2
        inputs = make_dummy_lidar_inputs(B)
        out = extractor.extract_topology(*inputs)
        for key in ('node_features', 'node_positions', 'edge_features',
                    'node_mask', 'edge_mask'):
            assert key in out, f"Missing key: {key}"

    def test_ego_position_preserved(self, extractor):
        """node_positions[:, 0, :] == ray_pos_w (ego position integrity)."""
        B = 4
        ray_hits, ray_pos, dyn_obs, ego_vel, target_pos = make_dummy_lidar_inputs(B)
        out = extractor.extract_topology(ray_hits, ray_pos, dyn_obs, ego_vel, target_pos)
        assert torch.allclose(out['node_positions'][:, 0, :], ray_pos, atol=1e-4), \
            "Ego position not preserved at index 0"

    def test_ego_always_valid(self, extractor):
        B = 4
        inputs = make_dummy_lidar_inputs(B)
        out = extractor.extract_topology(*inputs)
        assert out['node_mask'][:, 0].all(), "Ego node must always be valid"

    def test_no_nan_outputs(self, extractor):
        B = 4
        inputs = make_dummy_lidar_inputs(B)
        out = extractor.extract_topology(*inputs)
        for key in ('node_features', 'node_positions', 'edge_features'):
            masked_out = out[key]   # may contain zeros for padded nodes → OK
            valid_nodes = out['node_mask']
            # Check valid-node features only
            if key == 'node_features':
                vals = out[key][valid_nodes]        # (V, 18)
                assert not torch.isnan(vals).any(), f"NaN in {key} at valid nodes"

    def test_no_self_loops(self, extractor):
        B = 2
        inputs = make_dummy_lidar_inputs(B)
        out = extractor.extract_topology(*inputs)
        edge_mask = out['edge_mask']  # (B, N+1, N+1)
        for b in range(B):
            diag = torch.diagonal(edge_mask[b])
            assert not diag.any(), f"Self-loops in edge_mask at batch {b}"


# ---------------------------------------------------------------------------
# Stage 2: GraphTransformer
# ---------------------------------------------------------------------------

class TestGraphTransformerStage:

    def test_probs_valid_after_topo(self, extractor, policy):
        B = 4
        inputs = make_dummy_lidar_inputs(B)
        topo = extractor.extract_topology(*inputs)
        probs, h = policy(
            topo['node_features'],
            topo['edge_features'],
            topo['node_mask'],
            topo['edge_mask'],
        )
        assert not torch.isnan(probs).any(), "NaN in probs after full topo pipeline"
        assert torch.allclose(probs.sum(-1), torch.ones(B), atol=1e-5)

    def test_h_shape_after_topo(self, extractor, policy):
        B = 2
        inputs = make_dummy_lidar_inputs(B)
        topo = extractor.extract_topology(*inputs)
        _, h = policy(
            topo['node_features'],
            topo['edge_features'],
            topo['node_mask'],
            topo['edge_mask'],
        )
        assert h.shape == (B, N_TOT, 64)


# ---------------------------------------------------------------------------
# Stage 3: SafetyShield
# ---------------------------------------------------------------------------

class TestSafetyShieldStage:

    @staticmethod
    def _get_v_rl_from_policy(extractor, policy, B):
        inputs = make_dummy_lidar_inputs(B)
        topo = extractor.extract_topology(*inputs)
        probs, _ = policy(
            topo['node_features'],
            topo['edge_features'],
            topo['node_mask'],
            topo['edge_mask'],
        )
        # Simulate: pick highest-prob node, compute velocity toward it
        best_idx = probs.argmax(dim=-1) + 1   # +1 because ego is at 0
        ego_pos  = inputs[1]                   # ray_pos_w  (B, 3) tensor
        cand_pos = topo['node_positions'][torch.arange(B), best_idx]   # (B, 3)
        v_rl = (cand_pos - ego_pos)
        v_rl = v_rl / (v_rl.norm(dim=-1, keepdim=True).clamp(min=1e-6)) * 1.5
        dyn_obs = inputs[2]  # (B, 4, 10) tensor
        return v_rl, dyn_obs, ego_pos   # all torch tensors

    @staticmethod
    def _make_obs_tensor(dyn_obs: torch.Tensor, B: int) -> torch.Tensor:
        """Convert (B, N_dyn, 10) obstacle tensor → (B, N_dyn, 7) shield format."""
        obs_shield = torch.zeros(B, dyn_obs.shape[1], 7)
        obs_shield[:, :, :3] = dyn_obs[:, :, :3]   # positions
        obs_shield[:, :, 6]  = dyn_obs[:, :, 6]    # radius
        return obs_shield

    def test_full_pipeline_no_nan(self, extractor, policy, shield):
        B = 4
        v_rl, dyn_obs, ego_pos = self._get_v_rl_from_policy(extractor, policy, B)
        obs_shield = self._make_obs_tensor(dyn_obs, B)
        v_safe, intervention = shield.solve(v_rl, obs_shield, ego_pos)
        assert not torch.isnan(v_safe).any(), "NaN in v_safe from full pipeline"
        assert not torch.isnan(intervention).any()

    def test_intervention_shape_after_full_pipeline(self, extractor, policy, shield):
        B = 4
        v_rl, dyn_obs, ego_pos = self._get_v_rl_from_policy(extractor, policy, B)
        obs_shield = self._make_obs_tensor(dyn_obs, B)
        v_safe, intervention = shield.solve(v_rl, obs_shield, ego_pos)
        assert v_safe.shape == (B, 3)
        assert intervention.shape == (B,)


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------

class TestFullPipeline:

    def test_full_chain_shapes_and_validity(self, extractor, policy, shield):
        """Complete inference chain: LiDAR → Topo → Policy → QP → control."""
        B = 4
        ray_hits, ego_pos, dyn_obs, ego_vel, target_pos = make_dummy_lidar_inputs(B)

        # 1. Topology extraction
        topo = extractor.extract_topology(ray_hits, ego_pos, dyn_obs, ego_vel, target_pos)

        # 2. Graph policy
        probs, h = policy(
            topo['node_features'],
            topo['edge_features'],
            topo['node_mask'],
            topo['edge_mask'],
        )
        assert not torch.isnan(probs).any()

        # 3. Node → velocity
        best_idx = probs.argmax(dim=-1) + 1
        cand_pos = topo['node_positions'][torch.arange(B), best_idx]
        v_rl = cand_pos - ego_pos
        v_rl = v_rl / v_rl.norm(dim=-1, keepdim=True).clamp(min=1e-6) * 2.0

        # 4. QP safety shield
        obs_shield = torch.zeros(B, 4, 7)
        obs_shield[:, :, :3] = dyn_obs[:, :, :3]
        obs_shield[:, :, 6]  = dyn_obs[:, :, 6]
        v_safe, intervention = shield.solve(v_rl, obs_shield, ego_pos)

        assert v_safe.shape == (B, 3), f"Unexpected v_safe shape: {v_safe.shape}"
        assert intervention.shape == (B,)
        assert not torch.isnan(v_safe).any(), "NaN in final control velocity"
        assert (intervention >= 0).all(), "Intervention must be non-negative"

    def test_full_chain_performance_cpu(self, extractor, policy, shield):
        """
        Full inference chain should complete in < 5 s on CPU for B=4.
        (No performance pressure — just ensuring no infinite loop / deadlock.)
        """
        B = 4
        inputs = make_dummy_lidar_inputs(B)

        t0 = time.perf_counter()

        topo  = extractor.extract_topology(*inputs)
        probs, h = policy(
            topo['node_features'],
            topo['edge_features'],
            topo['node_mask'],
            topo['edge_mask'],
        )
        ray_hits, ego_pos, dyn_obs, ego_vel, target_pos = inputs
        best_idx = probs.argmax(dim=-1) + 1
        cand_pos = topo['node_positions'][torch.arange(B), best_idx]
        v_rl = (cand_pos - ego_pos)
        v_rl = v_rl / v_rl.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        obs_shield = torch.zeros(B, 4, 7)
        obs_shield[:, :, :3] = dyn_obs[:, :, :3]
        obs_shield[:, :, 6]  = dyn_obs[:, :, 6]
        shield.solve(v_rl, obs_shield, ego_pos)

        elapsed = time.perf_counter() - t0
        assert elapsed < 5.0, f"Full pipeline took {elapsed:.2f}s (> 5s limit)"

    def test_ego_position_integrity_through_pipeline(self, extractor, policy, shield):
        """
        node_positions[:, 0, :] must equal ray_pos_w after extraction.
        Ensures ego node is not displaced by any processing step.
        """
        B = 4
        ray_hits, ego_pos, dyn_obs, ego_vel, target_pos = make_dummy_lidar_inputs(B)
        topo = extractor.extract_topology(ray_hits, ego_pos, dyn_obs, ego_vel, target_pos)
        assert torch.allclose(topo['node_positions'][:, 0, :], ego_pos, atol=1e-4), \
            "Ego position corrupted in topology extraction"

    def test_v_safe_within_bounds(self, extractor, policy, shield):
        """v_safe speed must not exceed v_max (set in QP config)."""
        B = 4
        ray_hits, ego_pos, dyn_obs, ego_vel, target_pos = make_dummy_lidar_inputs(B)
        topo = extractor.extract_topology(ray_hits, ego_pos, dyn_obs, ego_vel, target_pos)
        probs, _ = policy(
            topo['node_features'], topo['edge_features'],
            topo['node_mask'], topo['edge_mask'],
        )
        best_idx = probs.argmax(dim=-1) + 1
        cand_pos = topo['node_positions'][torch.arange(B), best_idx]
        v_rl = (cand_pos - ego_pos)
        v_rl = v_rl / v_rl.norm(dim=-1, keepdim=True).clamp(min=1e-6) * 3.0

        obs_shield = torch.zeros(B, 4, 7)
        obs_shield[:, :, :3] = dyn_obs[:, :, :3]
        obs_shield[:, :, 6]  = 0.3
        v_safe, _ = shield.solve(v_rl, obs_shield, ego_pos)
        speeds = v_safe.norm(dim=-1)
        v_max  = make_qp_cfg().v_max
        assert (speeds <= v_max + 1e-3).all(), \
            f"v_safe exceeds v_max={v_max}: {speeds}"
