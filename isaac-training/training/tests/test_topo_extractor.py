"""Tests for TopoExtractor (pure PyTorch, no Isaac)."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

import torch
import pytest
from types import SimpleNamespace
from topo_extractor import TopoExtractor


@pytest.fixture
def cfg():
    return SimpleNamespace(
        max_nodes=10,
        node_feat_dim=18,
        edge_feat_dim=8,
        lidar_range=10.0,
        grid_size=50,       # small for test speed
        safe_radius=0.4,
        grad_threshold=0.5,
        nms_radius=0.5,
        k_neighbors=3,
        max_edge_length=2.0,
    )


@pytest.fixture
def extractor(cfg):
    return TopoExtractor(cfg)


def make_dummy_inputs(B=4, N_rays=144, N_dyn=8, device='cpu'):
    """Create synthetic LiDAR inputs that produce non-trivial distance fields."""
    torch.manual_seed(42)
    ego_pos    = torch.randn(B, 3) * 0.5
    ego_vel    = torch.randn(B, 3) * 0.3
    target_pos = ego_pos + torch.tensor([[5., 5., 0.]]).expand(B, -1)

    # Simulate ray hits at ~3-9 m range in a hemisphere
    angles = torch.linspace(0, 2 * 3.14159, N_rays).unsqueeze(0).expand(B, -1)
    ranges = 3.0 + 6.0 * torch.rand(B, N_rays)
    ray_hits_xy = torch.stack([
        ego_pos[:, 0:1] + ranges * torch.cos(angles),
        ego_pos[:, 1:2] + ranges * torch.sin(angles),
    ], dim=-1)
    ray_hits_z = ego_pos[:, 2:3].unsqueeze(1).expand(-1, N_rays, -1)
    ray_hits_w = torch.cat([ray_hits_xy, ray_hits_z], dim=-1)

    dyn_obs = torch.zeros(B, N_dyn, 10)  # all zeros = empty obstacles

    return ray_hits_w, ego_pos, dyn_obs, ego_vel, target_pos


# -------------------------------------------------------------------

class TestTopoExtractorShapes:

    def test_output_keys(self, extractor):
        ray_hits_w, ego_pos, dyn_obs, ego_vel, target_pos = make_dummy_inputs()
        out = extractor.extract_topology(ray_hits_w, ego_pos, dyn_obs, ego_vel, target_pos)
        assert 'node_features'  in out
        assert 'node_positions' in out
        assert 'edge_features'  in out
        assert 'node_mask'      in out
        assert 'edge_mask'      in out

    def test_node_features_shape(self, extractor, cfg):
        B = 4
        ray_hits_w, ego_pos, dyn_obs, ego_vel, target_pos = make_dummy_inputs(B=B)
        out = extractor.extract_topology(ray_hits_w, ego_pos, dyn_obs, ego_vel, target_pos)
        Np1 = cfg.max_nodes + 1
        assert out['node_features'].shape  == (B, Np1, cfg.node_feat_dim), \
            f"Expected ({B}, {Np1}, {cfg.node_feat_dim}), got {out['node_features'].shape}"

    def test_node_positions_shape(self, extractor, cfg):
        B = 4
        ray_hits_w, ego_pos, dyn_obs, ego_vel, target_pos = make_dummy_inputs(B=B)
        out = extractor.extract_topology(ray_hits_w, ego_pos, dyn_obs, ego_vel, target_pos)
        Np1 = cfg.max_nodes + 1
        assert out['node_positions'].shape == (B, Np1, 3)

    def test_edge_features_shape(self, extractor, cfg):
        B = 4
        ray_hits_w, ego_pos, dyn_obs, ego_vel, target_pos = make_dummy_inputs(B=B)
        out = extractor.extract_topology(ray_hits_w, ego_pos, dyn_obs, ego_vel, target_pos)
        Np1 = cfg.max_nodes + 1
        assert out['edge_features'].shape  == (B, Np1, Np1, cfg.edge_feat_dim)

    def test_node_mask_dtype_bool(self, extractor):
        ray_hits_w, ego_pos, dyn_obs, ego_vel, target_pos = make_dummy_inputs()
        out = extractor.extract_topology(ray_hits_w, ego_pos, dyn_obs, ego_vel, target_pos)
        assert out['node_mask'].dtype == torch.bool

    def test_edge_mask_dtype_bool(self, extractor):
        ray_hits_w, ego_pos, dyn_obs, ego_vel, target_pos = make_dummy_inputs()
        out = extractor.extract_topology(ray_hits_w, ego_pos, dyn_obs, ego_vel, target_pos)
        assert out['edge_mask'].dtype == torch.bool


class TestTopoExtractorSemantics:

    def test_ego_always_valid(self, extractor):
        """node_mask[:, 0] must always be True (ego node)."""
        ray_hits_w, ego_pos, dyn_obs, ego_vel, target_pos = make_dummy_inputs()
        out = extractor.extract_topology(ray_hits_w, ego_pos, dyn_obs, ego_vel, target_pos)
        assert out['node_mask'][:, 0].all(), "ego node (index 0) must always be valid"

    def test_ego_position_matches_ray_pos(self, extractor):
        """node_positions[:, 0] must equal ego_pos."""
        B = 4
        ray_hits_w, ego_pos, dyn_obs, ego_vel, target_pos = make_dummy_inputs(B=B)
        out = extractor.extract_topology(ray_hits_w, ego_pos, dyn_obs, ego_vel, target_pos)
        assert torch.allclose(out['node_positions'][:, 0, :], ego_pos, atol=1e-5), \
            "node_positions[:, 0] should equal ego_pos"

    def test_no_nan_in_output(self, extractor):
        ray_hits_w, ego_pos, dyn_obs, ego_vel, target_pos = make_dummy_inputs()
        out = extractor.extract_topology(ray_hits_w, ego_pos, dyn_obs, ego_vel, target_pos)
        for k, v in out.items():
            assert not torch.isnan(v.float()).any(), f"NaN found in {k}"

    def test_padded_nodes_zeroed_features(self, extractor, cfg):
        """Features of padded (invalid) nodes should be zero."""
        B = 2
        ray_hits_w, ego_pos, dyn_obs, ego_vel, target_pos = make_dummy_inputs(B=B)
        out = extractor.extract_topology(ray_hits_w, ego_pos, dyn_obs, ego_vel, target_pos)
        nf   = out['node_features']    # (B, N+1, 18)
        mask = out['node_mask']        # (B, N+1) bool
        # Padded nodes should have zero features
        padded = nf[~mask]
        assert (padded == 0).all(), "Padded node features should be zero"

    def test_edge_mask_symmetric(self, extractor):
        """Edge mask should be symmetric (undirected graph)."""
        ray_hits_w, ego_pos, dyn_obs, ego_vel, target_pos = make_dummy_inputs()
        out = extractor.extract_topology(ray_hits_w, ego_pos, dyn_obs, ego_vel, target_pos)
        em = out['edge_mask'].float()
        assert torch.allclose(em, em.transpose(1, 2)), "edge_mask should be symmetric"

    def test_no_self_loops(self, extractor, cfg):
        """Diagonal of edge_mask should be False."""
        ray_hits_w, ego_pos, dyn_obs, ego_vel, target_pos = make_dummy_inputs()
        out = extractor.extract_topology(ray_hits_w, ego_pos, dyn_obs, ego_vel, target_pos)
        em   = out['edge_mask']
        Np1  = cfg.max_nodes + 1
        diag = em[:, torch.arange(Np1), torch.arange(Np1)]
        assert not diag.any(), "edge_mask diagonal (self-loops) should be False"

    def test_nan_lidar_handled(self, extractor):
        """NaN in ray_hits_w should not crash and should produce valid output."""
        ray_hits_w, ego_pos, dyn_obs, ego_vel, target_pos = make_dummy_inputs()
        ray_hits_w[:, :10, :] = float('nan')   # inject NaN into first 10 rays
        out = extractor.extract_topology(ray_hits_w, ego_pos, dyn_obs, ego_vel, target_pos)
        for k, v in out.items():
            assert not torch.isnan(v.float()).any(), f"NaN in {k} after NaN LiDAR input"


class TestTopoExtractorBatchConsistency:

    def test_batch_size_1(self, extractor):
        ray_hits_w, ego_pos, dyn_obs, ego_vel, target_pos = make_dummy_inputs(B=1)
        out = extractor.extract_topology(ray_hits_w, ego_pos, dyn_obs, ego_vel, target_pos)
        assert out['node_features'].shape[0] == 1

    def test_batch_size_8(self, extractor):
        ray_hits_w, ego_pos, dyn_obs, ego_vel, target_pos = make_dummy_inputs(B=8)
        out = extractor.extract_topology(ray_hits_w, ego_pos, dyn_obs, ego_vel, target_pos)
        assert out['node_features'].shape[0] == 8
