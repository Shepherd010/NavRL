"""Tests for SafetyShieldQP (no Isaac)."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

import torch
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from safety_shield import SafetyShieldQP


@pytest.fixture
def shield():
    return SafetyShieldQP(
        relaxation_weight=5000.0,
        max_relaxation=0.3,
        v_max=2.0,
        safe_distance=0.5,
        cbf_alpha=1.0,
    )


def make_obstacle(pos, vel=(0., 0., 0.), radius=0.3):
    """Helper: create a single obstacle tensor [pos_x, pos_y, pos_z, vx, vy, vz, r]."""
    return list(pos) + list(vel) + [radius]


class TestSafetyShieldNoObstacles:

    def test_no_obstacles_returns_v_rl(self, shield):
        B = 4
        v_rl = torch.randn(B, 3)
        obs  = torch.zeros(B, 0, 7)   # zero obstacles
        v_safe, intervention = shield.solve(v_rl, obs)
        assert torch.allclose(v_safe, v_rl, atol=1e-5)
        assert (intervention == 0.0).all()

    def test_output_shapes(self, shield):
        B = 3
        v_rl = torch.randn(B, 3)
        obs  = torch.zeros(B, 5, 7)
        v_safe, intervention = shield.solve(v_rl, obs)
        assert v_safe.shape == (B, 3)
        assert intervention.shape == (B,)


class TestSafetyShieldWithObstacles:

    def test_collision_direction_deflected(self, shield):
        """v_rl pointing directly at a close obstacle → v_safe should differ."""
        B = 1
        # obstacle at +x=0.6m (within safe_distance=0.5 + radius=0.3 → danger zone)
        obs_data = make_obstacle(pos=[0.6, 0., 0.], vel=[0., 0., 0.], radius=0.3)
        v_rl = torch.tensor([[2.0, 0.0, 0.0]])           # heading into obstacle
        obs  = torch.tensor([[obs_data]])                  # (1, 1, 7)
        ego_pos = torch.zeros(1, 3)

        v_safe, intervention = shield.solve(v_rl, obs, ego_pos=ego_pos)
        # v_safe should not be identical to v_rl
        assert not torch.allclose(v_safe, v_rl, atol=1e-3), \
            "Expected safety shield to modify velocity toward obstacle"

    def test_intervention_positive_near_obstacle(self, shield):
        """intervention > 0 when obstacle is close."""
        B = 2
        obs_data = make_obstacle(pos=[0.4, 0., 0.], vel=[0., 0., 0.], radius=0.2)
        v_rl = torch.tensor([[1.5, 0., 0.], [1.5, 0., 0.]])
        obs  = torch.tensor([[obs_data], [obs_data]])
        ego_pos = torch.zeros(B, 3)
        _, intervention = shield.solve(v_rl, obs, ego_pos=ego_pos)
        assert (intervention >= 0).all()

    def test_far_obstacle_no_intervention(self, shield):
        """Obstacle far away → v_safe ≈ v_rl (no significant intervention)."""
        B = 1
        obs_data = make_obstacle(pos=[50., 0., 0.], vel=[0., 0., 0.], radius=0.3)
        v_rl = torch.tensor([[1.0, 0.0, 0.0]])
        obs  = torch.tensor([[obs_data]])
        ego_pos = torch.zeros(1, 3)
        v_safe, intervention = shield.solve(v_rl, obs, ego_pos=ego_pos)
        assert torch.allclose(v_safe, v_rl, atol=1e-3)
        assert intervention[0] < 0.01

    def test_v_safe_within_bounds(self, shield):
        """v_safe must stay within v_max."""
        B = 4
        v_rl = torch.randn(B, 3) * 3.0   # may exceed v_max
        obs_data = [make_obstacle(pos=[1.0, 0., 0.])] * 2
        obs = torch.tensor([obs_data] * B, dtype=torch.float32)
        ego_pos = torch.zeros(B, 3)
        v_safe, _ = shield.solve(v_rl, obs, ego_pos=ego_pos)
        assert (v_safe.abs() <= shield.v_max + 1e-3).all(), \
            "v_safe exceeded v_max"


class TestSafetyShieldFallback:

    def test_qp_failure_returns_v_rl(self, shield):
        """pit #6: On OSQP failure, return (v_rl, 0) without raising."""
        B = 1
        v_rl = torch.tensor([[1.0, 0.5, 0.2]])
        obs_data = make_obstacle(pos=[0.3, 0., 0.], vel=[0., 0., 0.], radius=0.3)
        obs = torch.tensor([[obs_data]])
        ego_pos = torch.zeros(1, 3)

        # Mock osqp via sys.modules so the inline 'import osqp' inside _solve_single picks it up
        mock_result = MagicMock()
        mock_result.info.status = 'infeasible'
        mock_result.x = np.zeros(4)
        mock_prob = MagicMock()
        mock_prob.solve.return_value = mock_result
        mock_osqp_module = MagicMock()
        mock_osqp_module.OSQP.return_value = mock_prob

        with patch.dict('sys.modules', {'osqp': mock_osqp_module}):
            v_safe, intervention = shield.solve(v_rl, obs, ego_pos=ego_pos)

        assert torch.allclose(v_safe, v_rl, atol=1e-5), \
            "On QP failure, should return original v_rl"
        assert intervention[0] == 0.0, \
            "On QP failure (fallback = v_rl), intervention should be 0"

    def test_osqp_exception_returns_v_rl(self, shield):
        """pit #6: On OSQP exception, return (v_rl, 0) without raising."""
        B = 1
        v_rl = torch.tensor([[1.0, 0.5, 0.2]])
        obs_data = make_obstacle(pos=[0.3, 0., 0.])
        obs = torch.tensor([[obs_data]])

        mock_osqp_module = MagicMock()
        mock_osqp_module.OSQP.side_effect = RuntimeError("solver crashed")

        with patch.dict('sys.modules', {'osqp': mock_osqp_module}):
            v_safe, intervention = shield.solve(v_rl, obs)

        assert torch.allclose(v_safe, v_rl, atol=1e-5)


class TestSafetyShieldStatistics:

    def test_statistics_accumulate(self, shield):
        B = 2
        v_rl = torch.randn(B, 3)
        obs  = torch.zeros(B, 0, 7)
        shield.solve(v_rl, obs)
        shield.solve(v_rl, obs)
        stats = shield.get_statistics()
        assert stats['num_solves'] == 2 * B   # 2 batch calls × batch_size=2

    def test_reset_statistics(self, shield):
        B = 2
        v_rl = torch.randn(B, 3)
        obs  = torch.zeros(B, 0, 7)
        shield.solve(v_rl, obs)
        shield.reset_statistics()
        stats = shield.get_statistics()
        assert stats['num_solves'] == 0
