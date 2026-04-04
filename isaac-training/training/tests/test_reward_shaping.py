"""Unit tests for RewardShaper."""

import pytest
import torch
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from reward_shaping import RewardShaper


B = 16  # batch size for tests


def _make_ctx(**overrides):
    """Build a minimal context dict with plausible tensor shapes."""
    device = "cpu"
    ctx = {
        "vel_w": torch.randn(B, 1, 3, device=device),
        "prev_vel_w": torch.randn(B, 1, 3, device=device),
        "target_dir_2d": torch.randn(B, 1, 3, device=device),
        "lidar_scan": torch.rand(B, 1, 36, 4, device=device) * 10.0,
        "lidar_range": 10.0,
        "drone_pos": torch.randn(B, 1, 3, device=device).abs() + 0.5,
        "height_range": torch.tensor([[0.5, 2.5]], device=device).expand(B, 1, 2),
        "collision": torch.zeros(B, 1, dtype=torch.bool, device=device),
        "reach_goal": torch.zeros(B, 1, dtype=torch.bool, device=device),
        "out_of_bounds_xy": torch.zeros(B, 1, dtype=torch.bool, device=device),
        "distance": torch.rand(B, 1, 1, device=device) * 20.0 + 1.0,
    }
    ctx.update(overrides)
    return ctx


class TestRewardShaperOutputShape:
    @pytest.mark.parametrize("preset", ["stable_flight", "goal_navigation", "full_navigation"])
    def test_shape(self, preset):
        shaper = RewardShaper(preset=preset, speed_target=2.5, speed_sigma=1.0)
        reward = shaper.compute(_make_ctx())
        assert reward.shape == (B, 1), f"Expected (B,1), got {reward.shape}"

    @pytest.mark.parametrize("preset", ["stable_flight", "goal_navigation", "full_navigation"])
    def test_finite(self, preset):
        shaper = RewardShaper(preset=preset)
        reward = shaper.compute(_make_ctx())
        assert torch.isfinite(reward).all(), "Reward contains non-finite values"


class TestRewardShaperReachGoal:
    def test_reach_goal_bonus(self):
        shaper = RewardShaper(preset="goal_navigation")
        ctx_no = _make_ctx(reach_goal=torch.zeros(B, 1, dtype=torch.bool))
        ctx_yes = _make_ctx(reach_goal=torch.ones(B, 1, dtype=torch.bool))
        r_no = shaper.compute(ctx_no)
        r_yes = shaper.compute(ctx_yes)
        # reach_goal adds 500, so average should be much higher
        assert (r_yes.mean() - r_no.mean()) > 400.0


class TestRewardShaperCollision:
    def test_collision_penalty(self):
        shaper = RewardShaper(preset="stable_flight")
        ctx_safe = _make_ctx(collision=torch.zeros(B, 1, dtype=torch.bool))
        ctx_crash = _make_ctx(collision=torch.ones(B, 1, dtype=torch.bool))
        r_safe = shaper.compute(ctx_safe)
        r_crash = shaper.compute(ctx_crash)
        assert (r_safe.mean() - r_crash.mean()) > 150.0


class TestProgressLocal:
    def test_positive_when_moving_toward_goal(self):
        """Flying directly toward goal should yield positive progress."""
        direction = torch.zeros(B, 1, 3)
        direction[:, :, 0] = 1.0  # goal is +x
        vel = torch.zeros(B, 1, 3)
        vel[:, :, 0] = 2.0  # flying +x at 2 m/s
        prog = RewardShaper._progress_local(vel, direction)
        assert (prog > 0).all()

    def test_negative_when_retreating(self):
        direction = torch.zeros(B, 1, 3)
        direction[:, :, 0] = 1.0
        vel = torch.zeros(B, 1, 3)
        vel[:, :, 0] = -2.0  # flying away
        prog = RewardShaper._progress_local(vel, direction)
        assert (prog < 0).all()


class TestSpeedBand:
    def test_max_at_target(self):
        vel = torch.zeros(B, 1, 3)
        vel[:, :, 0] = 2.5  # exactly target speed
        r = RewardShaper._speed_band(vel, target=2.5, sigma=1.0)
        assert (r > 0.99).all()

    def test_lower_away_from_target(self):
        vel_on = torch.zeros(B, 1, 3)
        vel_on[:, :, 0] = 2.5
        vel_off = torch.zeros(B, 1, 3)
        vel_off[:, :, 0] = 0.0
        r_on = RewardShaper._speed_band(vel_on, target=2.5, sigma=1.0)
        r_off = RewardShaper._speed_band(vel_off, target=2.5, sigma=1.0)
        assert (r_on > r_off).all()


class TestUpdateConfig:
    def test_switch_preset(self):
        shaper = RewardShaper(preset="stable_flight")
        shaper.update_config(preset="goal_navigation", speed_target=3.0, speed_sigma=1.0)
        assert shaper.preset == "goal_navigation"
        assert shaper.speed_target == 3.0

    def test_unknown_preset_raises(self):
        shaper = RewardShaper(preset="nonexistent")
        with pytest.raises(ValueError, match="Unknown reward preset"):
            shaper.compute(_make_ctx())


class TestFullNavigationWithDynObs:
    def test_with_dynamic_obstacles(self):
        shaper = RewardShaper(preset="full_navigation")
        ctx = _make_ctx()
        ctx["closest_dyn_obs_distance_reward"] = torch.rand(B, 5) * 5.0 + 0.5
        reward = shaper.compute(ctx)
        assert reward.shape == (B, 1)
        assert torch.isfinite(reward).all()
