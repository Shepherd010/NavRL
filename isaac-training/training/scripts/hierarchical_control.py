"""
HierarchicalController: multi-rate planner/controller.

Pure PyTorch (+ NumPy for stats). No Isaac Sim dependency.

High-level planning  : 10 Hz  (graph inference, once every freq_ratio steps)
Low-level control    : 62.5 Hz (PID tracking, every step)

freq_ratio = int(round(62.5 / 10.0)) = 6   ← pit #4: NOT 6.25

Public methods:
    reset(batch_size, device)          full reset (e.g. beginning of training)
    reset_partial(env_ids)             per-episode reset, call from _reset_idx()  ← pit #5
    step(ego_pos, ego_vel, [v_hl])     one 62.5Hz tick; provide v_hl only on high-level steps
    is_high_level_step() -> bool       query whether current tick is a high-level update
"""

import numpy as np
import torch
from typing import Optional, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class HierarchicalController:

    def __init__(
        self,
        high_freq:    float = 62.5,
        low_freq:     float = 10.0,
        pid_kp:       float = 2.0,
        pid_kd:       float = 0.8,
        pid_ki:       float = 0.05,
        goal_horizon: float = 0.1,
        integral_limit: float = 1.0,
    ):
        self.high_freq  = high_freq
        self.low_freq   = low_freq
        self.dt_high    = 1.0 / high_freq  # 16 ms per sim step

        # pit #4: freq_ratio must be int(round(...)), not 6.25
        self.freq_ratio = int(round(high_freq / low_freq))  # 6

        self.pid_gains  = {'kp': pid_kp, 'kd': pid_kd, 'ki': pid_ki}
        self.goal_horizon  = goal_horizon
        self.integral_limit = integral_limit

        # Runtime state (initialised in reset())
        self.step_counter:      int                        = 0
        self.intermediate_goal: Optional[torch.Tensor]     = None  # (B, 3)
        self.last_error:        Optional[torch.Tensor]     = None  # (B, 3)
        self.error_integral:    Optional[torch.Tensor]     = None  # (B, 3)

        # Statistics
        self.num_high_updates = 0
        self.num_low_updates  = 0
        self.tracking_errors: list = []

    # ------------------------------------------------------------------
    # Reset helpers
    # ------------------------------------------------------------------

    def reset(self, batch_size: int, device: torch.device):
        """Full reset — call once before training loop starts."""
        self.step_counter     = 0
        self.intermediate_goal = None
        self.last_error        = torch.zeros(batch_size, 3, device=device)
        self.error_integral    = torch.zeros(batch_size, 3, device=device)
        self.num_high_updates  = 0
        self.num_low_updates   = 0
        self.tracking_errors   = []

    def reset_partial(self, env_ids: torch.Tensor):
        """
        Per-episode reset called from NavigationEnv._reset_idx(env_ids).

        pit #5: This MUST be in _reset_idx, NOT in the train.py loop,
        because SyncDataCollector encapsulates env.step and doesn't expose
        per-env done flags to the outer loop.

        Clears PID state for finished envs; other envs are unaffected.
        step_counter is global and intentionally not reset per-env.
        """
        if self.error_integral is None or self.last_error is None:
            return  # not yet initialised
        if env_ids.numel() == 0:
            return
        self.error_integral[env_ids] = 0.0
        self.last_error[env_ids]     = 0.0
        # intermediate_goal will be overwritten on the next high-level tick

    # ------------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------------

    def step(
        self,
        ego_pos:           torch.Tensor,               # (B, 3)
        ego_vel:           torch.Tensor,               # (B, 3)
        high_level_velocity: Optional[torch.Tensor] = None,  # (B, 3)
    ) -> Tuple[torch.Tensor, Dict]:
        """
        One 62.5 Hz control tick.

        On high-level update ticks (step_counter % freq_ratio == 0):
            high_level_velocity must be provided.
        On other ticks:
            high_level_velocity is ignored; PID tracks existing goal.

        Returns:
            control_velocity : (B, 3)
            info             : dict with diagnostics
        """
        B      = ego_pos.shape[0]
        device = ego_pos.device

        # Lazy init (handles first call without explicit reset())
        if self.last_error is None:
            self.last_error     = torch.zeros(B, 3, device=device)
            self.error_integral = torch.zeros(B, 3, device=device)

        is_hl = self.is_high_level_step()

        if is_hl:
            if high_level_velocity is None:
                raise ValueError(
                    "high_level_velocity must be provided at high-level update ticks "
                    f"(tick {self.step_counter}, freq_ratio={self.freq_ratio})"
                )
            self.intermediate_goal = ego_pos + high_level_velocity * self.goal_horizon
            self.num_high_updates += 1
        else:
            if self.intermediate_goal is None:
                # Before first high-level tick: hover
                self.intermediate_goal = ego_pos.clone()

        # PID control
        ctrl_vel = self._pid(ego_pos, ego_vel)

        # Tracking error
        track_err = (ego_pos - self.intermediate_goal).norm(dim=-1)
        self.tracking_errors.append(track_err.mean().item())
        if len(self.tracking_errors) > 1000:
            self.tracking_errors = self.tracking_errors[-1000:]

        self.step_counter    += 1
        self.num_low_updates += 1

        info = {
            'is_high_update':     is_hl,
            'tracking_error':     track_err,
            'intermediate_goal':  self.intermediate_goal,
            'step_counter':       self.step_counter,
        }
        return ctrl_vel, info

    def is_high_level_step(self) -> bool:
        return (self.step_counter % self.freq_ratio) == 0

    # ------------------------------------------------------------------
    # PID
    # ------------------------------------------------------------------

    def _pid(
        self,
        ego_pos: torch.Tensor,   # (B, 3)
        ego_vel: torch.Tensor,   # (B, 3)
    ) -> torch.Tensor:

        error = self.intermediate_goal - ego_pos   # (B, 3)

        if self.last_error is None:
            d_error = torch.zeros_like(error)
        else:
            d_error = (error - self.last_error) / self.dt_high

        self.error_integral = self.error_integral + error * self.dt_high
        self.error_integral = self.error_integral.clamp(
            -self.integral_limit, self.integral_limit
        )

        kp = self.pid_gains['kp']
        kd = self.pid_gains['kd']
        ki = self.pid_gains['ki']
        ctrl = kp * error + kd * d_error + ki * self.error_integral

        self.last_error = error.clone()
        return ctrl

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> dict:
        realized = (self.num_low_updates / max(self.num_high_updates, 1))
        return {
            'freq_ratio':          self.freq_ratio,
            'freq_ratio_realized': realized,
            'num_high_updates':    self.num_high_updates,
            'num_low_updates':     self.num_low_updates,
            'avg_tracking_error':  float(np.mean(self.tracking_errors))
                                   if self.tracking_errors else 0.0,
        }
