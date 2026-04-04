"""Privilege-free reward shaping presets for curriculum training.

Three presets:
  1. stable_flight     — speed band + heading + height + smoothness
  2. goal_navigation   — above + velocity-projected progress + reach bonus
  3. full_navigation   — above + LiDAR safety + dynamic obstacle safety

Design principle: **NO global distance to goal** is used.  All signals derive
from quantities available on a real robot (velocity, direction vector from GPS,
LiDAR scan, dynamic obstacle detector).

No Isaac Sim dependency — operates on plain tensors.
"""

from __future__ import annotations
import torch
from typing import Dict, Any, Optional


class RewardShaper:
    """Compute per-step reward given a reward preset name and env tensors."""

    def __init__(self, preset: str, speed_target: float = 2.5, speed_sigma: float = 1.0):
        self.preset = preset
        self.speed_target = speed_target
        self.speed_sigma = speed_sigma

    def update_config(self, preset: str, speed_target: float, speed_sigma: float) -> None:
        self.preset = preset
        self.speed_target = speed_target
        self.speed_sigma = speed_sigma

    # ------------------------------------------------------------------
    # Main entry — dispatches to the correct preset
    # ------------------------------------------------------------------

    def compute(self, ctx: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute scalar reward (B, 1).

        ``ctx`` must contain the following keys produced by env._compute_state_and_obs:
            vel_w           (B, 1, 3)  world velocity
            prev_vel_w      (B, 1, 3)  previous step velocity
            target_dir_2d   (B, 1, 3)  unnormalised start→goal direction (z zeroed)
            lidar_scan      (B, 1, H, V) lidar scan (range - distance)
            lidar_range     float       lidar max range
            drone_pos       (B, 1, 3)  world position
            height_range    (B, 1, 2)  [min_z, max_z] allowed band
            collision       (B, 1) bool
            reach_goal      (B, 1) bool
            out_of_bounds_xy (B, 1) bool
            distance        (B, 1, 1)  current distance to goal (for reach_goal only)
        Optional:
            closest_dyn_obs_distance_reward  (B, N) obstacle surface distances
            dyn_obs_range_mask               (B, N) mask for out-of-range obs
        """
        if self.preset == "stable_flight":
            return self._stable_flight(ctx)
        elif self.preset == "goal_navigation":
            return self._goal_navigation(ctx)
        elif self.preset == "full_navigation":
            return self._full_navigation(ctx)
        else:
            raise ValueError(f"Unknown reward preset: {self.preset}")

    # ------------------------------------------------------------------
    # Shared reward components
    # ------------------------------------------------------------------

    @staticmethod
    def _speed_band(vel_w: torch.Tensor, target: float, sigma: float) -> torch.Tensor:
        """Gaussian reward centred on target speed.  Returns (B, 1) in [0, 1]."""
        speed = vel_w[..., :3].norm(dim=-1, keepdim=True)       # (B, 1, 1) or (B, 1)
        if speed.dim() == 3:
            speed = speed.squeeze(1)                             # (B, 1)
        return torch.exp(-((speed - target) ** 2) / (2 * sigma ** 2))

    @staticmethod
    def _heading_reward(vel_w: torch.Tensor, target_dir_2d: torch.Tensor) -> torch.Tensor:
        """Cosine similarity between velocity and target direction. (B, 1) in [-1, 1].

        Uses only the direction observation — no privileged global distance.
        """
        vel_2d = vel_w[..., :2]                                  # (B, 1, 2)
        dir_2d = target_dir_2d[..., :2]                          # (B, 1, 2)
        vel_norm = vel_2d.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        dir_norm = dir_2d.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        cos_sim = (vel_2d * dir_2d).sum(dim=-1, keepdim=True) / (vel_norm * dir_norm)
        return cos_sim.squeeze(1)                                # (B, 1)

    @staticmethod
    def _progress_local(vel_w: torch.Tensor, target_dir_2d: torch.Tensor) -> torch.Tensor:
        """Velocity projected onto goal direction — privilege-free progress signal.

        Equivalent to 'how many metres did the drone move toward the goal this step'
        (before multiplying by dt, but dt is constant so it's a linear scale).
        Positive when flying toward goal, negative when retreating.
        """
        dir_2d = target_dir_2d[..., :2]                          # (B, 1, 2)
        dir_norm = dir_2d.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        d_hat = dir_2d / dir_norm                                # unit vector (B, 1, 2)
        vel_2d = vel_w[..., :2]                                  # (B, 1, 2)
        proj = (vel_2d * d_hat).sum(dim=-1, keepdim=True)       # (B, 1, 1)
        return proj.squeeze(1)                                   # (B, 1)

    @staticmethod
    def _smoothness(vel_w: torch.Tensor, prev_vel_w: torch.Tensor) -> torch.Tensor:
        """Jerk penalty (B, 1)."""
        diff = (vel_w[..., :3] - prev_vel_w[..., :3]).norm(dim=-1)  # (B, 1) or (B,)
        if diff.dim() == 1:
            diff = diff.unsqueeze(-1)
        return diff

    @staticmethod
    def _height_penalty(drone_pos: torch.Tensor, height_range: torch.Tensor) -> torch.Tensor:
        """Quadratic penalty when outside allowed height band. (B, 1)."""
        z = drone_pos[..., 2]          # (B, 1)
        z_lo = height_range[..., 0]    # (B, 1)
        z_hi = height_range[..., 1]    # (B, 1)
        penalty = torch.zeros_like(z)
        above = z > (z_hi + 0.2)
        below = z < (z_lo - 0.2)
        penalty[above] = ((z - z_hi - 0.2) ** 2)[above]
        penalty[below] = ((z_lo - 0.2 - z) ** 2)[below]
        return penalty

    # ------------------------------------------------------------------
    # Presets
    # ------------------------------------------------------------------

    def _stable_flight(self, ctx: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Stage 1: learn to fly forward stably at desired speed."""
        r_speed = self._speed_band(ctx["vel_w"], self.speed_target, self.speed_sigma)
        r_heading = self._heading_reward(ctx["vel_w"], ctx["target_dir_2d"])
        r_smooth = self._smoothness(ctx["vel_w"], ctx["prev_vel_w"])
        p_height = self._height_penalty(ctx["drone_pos"], ctx["height_range"])
        collision = ctx["collision"]
        oob = ctx["out_of_bounds_xy"]

        reward = (
              r_speed * 5.0                     # speed band: 0–5
            + r_heading * 8.0                   # heading alignment: −8 to +8
            + 1.0                               # survival bonus
            - r_smooth * 2.0                    # jerk penalty
            - p_height * 20.0                   # height OOB
            - collision.float() * 200.0         # collision
            - oob.float() * 200.0               # horizontal OOB
        )
        return reward

    def _goal_navigation(self, ctx: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Stage 2: navigate to target using privilege-free progress signal."""
        r_speed = self._speed_band(ctx["vel_w"], self.speed_target, self.speed_sigma)
        r_heading = self._heading_reward(ctx["vel_w"], ctx["target_dir_2d"])
        r_progress = self._progress_local(ctx["vel_w"], ctx["target_dir_2d"])
        r_smooth = self._smoothness(ctx["vel_w"], ctx["prev_vel_w"])
        p_height = self._height_penalty(ctx["drone_pos"], ctx["height_range"])
        collision = ctx["collision"]
        oob = ctx["out_of_bounds_xy"]
        reach_goal = ctx["reach_goal"]

        reward = (
              r_progress * 10.0                 # velocity-projected progress: ±50
            + r_speed * 3.0                     # speed band: 0–3
            + r_heading * 5.0                   # heading: −5 to +5
            + reach_goal.float() * 500.0        # terminal success
            + 1.0                               # survival
            - r_smooth * 2.0                    # jerk
            - p_height * 20.0                   # height OOB
            - collision.float() * 200.0         # collision
            - oob.float() * 200.0               # horizontal OOB
        )
        return reward

    def _full_navigation(self, ctx: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Stage 3: full obstacle avoidance navigation."""
        r_speed = self._speed_band(ctx["vel_w"], self.speed_target, self.speed_sigma)
        r_heading = self._heading_reward(ctx["vel_w"], ctx["target_dir_2d"])
        r_progress = self._progress_local(ctx["vel_w"], ctx["target_dir_2d"])
        r_smooth = self._smoothness(ctx["vel_w"], ctx["prev_vel_w"])
        p_height = self._height_penalty(ctx["drone_pos"], ctx["height_range"])
        collision = ctx["collision"]
        oob = ctx["out_of_bounds_xy"]
        reach_goal = ctx["reach_goal"]

        # LiDAR static safety: log(min_clearance) — same formula as original but as bonus
        lidar_scan = ctx["lidar_scan"]
        lidar_range = ctx["lidar_range"]
        r_safety_static = torch.log(
            (lidar_range - lidar_scan).clamp(min=1e-6, max=lidar_range)
        ).mean(dim=(2, 3))  # (B, 1)

        # Dynamic obstacle safety
        r_safety_dynamic = torch.zeros_like(r_safety_static)
        if "closest_dyn_obs_distance_reward" in ctx:
            dyn_dist = ctx["closest_dyn_obs_distance_reward"]
            r_safety_dynamic = torch.log(
                dyn_dist.clamp(min=1e-6, max=lidar_range)
            ).mean(dim=-1, keepdim=True)

        reward = (
              r_progress * 10.0                 # velocity-projected progress
            + r_speed * 3.0                     # speed band
            + r_heading * 5.0                   # heading
            + reach_goal.float() * 500.0        # terminal success
            + 1.0                               # survival
            + r_safety_static * 1.0             # static obstacle awareness
            + r_safety_dynamic * 1.0            # dynamic obstacle awareness
            - r_smooth * 2.0                    # jerk
            - p_height * 20.0                   # height OOB
            - collision.float() * 200.0         # collision
            - oob.float() * 200.0               # horizontal OOB
        )
        return reward
