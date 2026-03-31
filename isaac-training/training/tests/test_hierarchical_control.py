"""Tests for HierarchicalController (no Isaac)."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

import torch
import pytest
from hierarchical_control import HierarchicalController


@pytest.fixture
def ctrl():
    return HierarchicalController(
        high_freq=62.5,
        low_freq=10.0,
        pid_kp=2.0,
        pid_kd=0.8,
        pid_ki=0.05,
        goal_horizon=0.1,
    )


class TestFreqRatio:

    def test_freq_ratio_is_6(self, ctrl):
        """pit #4: freq_ratio must be 6, NOT 6.25 or 7."""
        assert ctrl.freq_ratio == 6, \
            f"freq_ratio should be int(round(62.5/10.0))=6, got {ctrl.freq_ratio}"

    def test_first_step_is_high_level(self, ctrl):
        """Step 0 must always be a high-level update."""
        assert ctrl.is_high_level_step() is True

    def test_high_level_steps_schedule(self, ctrl):
        """Ticks 0,6,12,18,... are high-level; 1..5, 7..11,... are low-level."""
        high_ticks = []
        for i in range(30):
            if ctrl.is_high_level_step():
                high_ticks.append(i)
            # advance counter
            ctrl.step_counter += 1
        assert high_ticks == [0, 6, 12, 18, 24]


class TestStepInterface:

    def test_step_output_shape(self, ctrl):
        B = 4
        ego_pos = torch.zeros(B, 3)
        ego_vel = torch.zeros(B, 3)
        v_hl    = torch.ones(B, 3)
        ctrl_vel, info = ctrl.step(ego_pos, ego_vel, high_level_velocity=v_hl)
        assert ctrl_vel.shape == (B, 3)

    def test_step_without_hl_on_non_hl_tick(self, ctrl):
        """After first HL tick, non-HL ticks don't need high_level_velocity."""
        B = 2
        ego_pos = torch.zeros(B, 3)
        ego_vel = torch.zeros(B, 3)
        v_hl    = torch.ones(B, 3)
        # Tick 0 (HL)
        ctrl.step(ego_pos, ego_vel, high_level_velocity=v_hl)
        # Ticks 1..5 (non-HL)
        for _ in range(5):
            ctrl_vel, _ = ctrl.step(ego_pos, ego_vel)   # no v_hl → should be fine
            assert ctrl_vel.shape == (B, 3)

    def test_step_missing_hl_velocity_raises(self, ctrl):
        """On tick 0 (HL tick), missing high_level_velocity must raise ValueError."""
        B = 2
        ego_pos = torch.zeros(B, 3)
        ego_vel = torch.zeros(B, 3)
        with pytest.raises(ValueError):
            ctrl.step(ego_pos, ego_vel, high_level_velocity=None)

    def test_info_keys(self, ctrl):
        B = 2
        ego_pos = torch.zeros(B, 3)
        ego_vel = torch.zeros(B, 3)
        v_hl    = torch.ones(B, 3)
        _, info = ctrl.step(ego_pos, ego_vel, high_level_velocity=v_hl)
        for key in ('is_high_update', 'tracking_error', 'intermediate_goal', 'step_counter'):
            assert key in info

    def test_intermediate_goal_tracks_velocity(self, ctrl):
        """intermediate_goal = ego_pos + v_hl * goal_horizon."""
        B = 1
        ego_pos = torch.zeros(B, 3)
        ego_vel = torch.zeros(B, 3)
        v_hl    = torch.tensor([[1.0, 0.0, 0.0]])
        _, info = ctrl.step(ego_pos, ego_vel, high_level_velocity=v_hl)
        expected_goal = ego_pos + v_hl * ctrl.goal_horizon
        assert torch.allclose(info['intermediate_goal'], expected_goal, atol=1e-5)


class TestResetPartial:

    def test_reset_partial_clears_target_envs(self, ctrl):
        """
        pit #5: reset_partial(env_ids) must zero PID state ONLY for specified envs.
        """
        B = 4
        device = 'cpu'
        ctrl.reset(B, device)

        # Inject some non-zero integral / error
        ctrl.error_integral = torch.ones(B, 3)
        ctrl.last_error     = torch.ones(B, 3)

        # Reset only envs 0 and 2
        env_ids = torch.tensor([0, 2])
        ctrl.reset_partial(env_ids)

        assert torch.allclose(ctrl.error_integral[0], torch.zeros(3)), "env 0 integral not cleared"
        assert torch.allclose(ctrl.error_integral[2], torch.zeros(3)), "env 2 integral not cleared"
        # Envs 1 and 3 must be untouched
        assert torch.allclose(ctrl.error_integral[1], torch.ones(3)),  "env 1 should be unchanged"
        assert torch.allclose(ctrl.error_integral[3], torch.ones(3)),  "env 3 should be unchanged"

    def test_reset_partial_empty_ids(self, ctrl):
        """Empty env_ids should not crash."""
        B = 4
        ctrl.reset(B, 'cpu')
        ctrl.error_integral = torch.ones(B, 3)
        ctrl.reset_partial(torch.tensor([], dtype=torch.long))
        assert torch.allclose(ctrl.error_integral, torch.ones(B, 3))

    def test_reset_partial_before_init(self, ctrl):
        """reset_partial before reset() should not crash (lazy init guard)."""
        ctrl.reset_partial(torch.tensor([0, 1]))  # should silently return


class TestPIDControl:

    def test_control_towards_goal(self, ctrl):
        """PID output should roughly point from ego toward goal."""
        B = 1
        ctrl.reset(B, 'cpu')
        ego_pos = torch.zeros(B, 3)
        ego_vel = torch.zeros(B, 3)
        v_hl    = torch.tensor([[1.0, 0.0, 0.0]])

        # HL tick to set goal
        ctrl.step(ego_pos, ego_vel, high_level_velocity=v_hl)

        # Move ego, check PID
        ego_pos2 = torch.zeros(B, 3)  # still at origin
        ctrl_vel, _ = ctrl.step(ego_pos2, ego_vel)
        # Should push in +x direction (toward intermediate_goal at [0.1, 0, 0])
        assert ctrl_vel[0, 0] > 0, "PID should push toward +x goal"

    def test_integral_antiwindup(self, ctrl):
        """Integral must be clamped to integral_limit."""
        B = 1
        ctrl.reset(B, 'cpu')
        v_hl = torch.ones(B, 3) * 2.0  # goal far away
        ego_pos = torch.zeros(B, 3)
        ego_vel = torch.zeros(B, 3)

        # Run many steps to accumulate integral
        ctrl.step(ego_pos, ego_vel, high_level_velocity=v_hl)
        for _ in range(200):
            if ctrl.is_high_level_step():
                ctrl.step(ego_pos, ego_vel, high_level_velocity=v_hl)
            else:
                ctrl.step(ego_pos, ego_vel)

        max_integral = ctrl.error_integral.abs().max().item()
        assert max_integral <= ctrl.integral_limit + 1e-5, \
            f"Integral exceeded limit: {max_integral} > {ctrl.integral_limit}"


class TestStatistics:

    def test_statistics_track_updates(self, ctrl):
        B = 2
        ctrl.reset(B, 'cpu')
        ego_pos = torch.zeros(B, 3)
        ego_vel = torch.zeros(B, 3)
        v_hl    = torch.ones(B, 3)

        # Run 12 steps: should give 2 HL updates (at 0 and 6)
        for i in range(12):
            if ctrl.is_high_level_step():
                ctrl.step(ego_pos, ego_vel, high_level_velocity=v_hl)
            else:
                ctrl.step(ego_pos, ego_vel)

        stats = ctrl.get_statistics()
        assert stats['freq_ratio'] == 6
        assert stats['num_high_updates'] == 2
        assert stats['num_low_updates']  == 12
