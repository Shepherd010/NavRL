"""
SafetyShieldQP: soft-constraint QP safety filter.

Pure Python / NumPy / OSQP. No Isaac Sim dependency.

Input:
    v_rl       : (B, 3)          policy velocity command (world frame)
    obstacles  : (B, N_obs, 7)   [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, radius]
    ego_pos    : (B, 3) optional ego world position

Output:
    v_safe     : (B, 3)          safe velocity (world frame, torch Tensor)
    intervention : (B,)          ||v_safe - v_rl||  (torch Tensor)

Pit #6: On QP failure, return (v_rl, 0) — never raise.
"""

import time
import numpy as np
import scipy.sparse as sp
import torch
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class SafetyShieldQP:

    def __init__(
        self,
        relaxation_weight: float = 5000.0,
        max_relaxation:    float = 0.3,
        v_max:             float = 2.0,
        safe_distance:     float = 0.5,
        cbf_alpha:         float = 1.0,
        solver_settings:   dict  = None,
    ):
        self.relaxation_weight = relaxation_weight
        self.max_relaxation    = max_relaxation
        self.v_max             = v_max
        self.safe_distance     = safe_distance
        self.cbf_alpha         = cbf_alpha

        self.solver_settings = solver_settings or {
            'verbose':        False,
            'eps_abs':        1e-4,
            'eps_rel':        1e-4,
            'max_iter':       2000,
            'polish':         True,
            'adaptive_rho':   True,
        }

        # Statistics
        self.num_solves     = 0
        self.num_failures   = 0
        self.total_time     = 0.0
        self.intervention_history = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def solve(
        self,
        v_rl:      torch.Tensor,            # (B, 3)
        obstacles: torch.Tensor,            # (B, N_obs, 7)
        ego_pos:   Optional[torch.Tensor] = None,  # (B, 3)
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        B      = v_rl.shape[0]
        device = v_rl.device

        v_rl_np  = v_rl.detach().cpu().numpy()           # (B, 3)
        obs_np   = obstacles.detach().cpu().numpy()      # (B, N_obs, 7)
        ego_np   = ego_pos.detach().cpu().numpy() if ego_pos is not None else None

        v_safe_list = []
        intv_list   = []
        for b in range(B):
            ego_b = ego_np[b] if ego_np is not None else None
            vs, intv = self._solve_single(v_rl_np[b], obs_np[b], ego_b)
            v_safe_list.append(vs)
            intv_list.append(intv)

        v_safe      = torch.from_numpy(np.stack(v_safe_list, axis=0)).float().to(device)
        intervention = torch.tensor(intv_list, dtype=torch.float32, device=device)
        return v_safe, intervention

    # ------------------------------------------------------------------
    # Single QP
    # ------------------------------------------------------------------

    def _solve_single(
        self,
        v_rl: np.ndarray,         # (3,)
        obstacles: np.ndarray,    # (N_obs, 7)
        ego_pos: Optional[np.ndarray] = None,  # (3,)
    ) -> Tuple[np.ndarray, float]:

        import osqp  # import here so module can be loaded without osqp (for mocking)
        t0 = time.perf_counter()

        # Filter nearby obstacles
        if ego_pos is not None:
            dists = np.linalg.norm(obstacles[:, :3] - ego_pos, axis=1)
            obstacles = obstacles[dists < self.safe_distance * 5]

        n_obs = obstacles.shape[0]

        # No obstacles → return policy velocity unchanged
        if n_obs == 0:
            self.num_solves += 1
            return v_rl.copy(), 0.0

        n_vars = 3 + n_obs

        # ---- Objective ----
        P_diag = np.concatenate([
            np.ones(3) * 2.0,
            np.ones(n_obs) * 2.0 * self.relaxation_weight,
        ])
        P = sp.diags(P_diag, format='csc')
        q = np.concatenate([-2.0 * v_rl, np.zeros(n_obs)])

        # ---- Constraints ----
        A_rows, l_list, u_list = [], [], []

        for i in range(n_obs):
            pos    = obstacles[i, :3]
            vel_o  = obstacles[i, 3:6]
            radius = obstacles[i, 6]

            ref = ego_pos if ego_pos is not None else -pos
            rel = pos - (ref if ego_pos is not None else np.zeros(3))
            d   = np.linalg.norm(rel)

            if d < 1e-3:
                continue  # degenerate, skip

            n_hat = rel / d

            h_i   = d - self.safe_distance - radius
            # Upper-bound CBF: n_hat·v ≤ cbf_alpha*h + n_hat·vel_obs + slack
            # (n_hat·v - slack ≤ u_cbf; slack≥0 softens the constraint)
            u_cbf = np.dot(n_hat, vel_o) + self.cbf_alpha * h_i

            row = np.zeros(n_vars)
            row[:3]    = n_hat
            row[3 + i] = -1.0   # slack relaxes upper bound: n_hat·v - slack ≤ u_cbf
            A_rows.append(row)
            l_list.append(-1e10)
            u_list.append(u_cbf)

        # Velocity box
        for d in range(3):
            row = np.zeros(n_vars); row[d] = 1.0
            A_rows.append(row)
            l_list.append(-self.v_max)
            u_list.append(self.v_max)

        # Slack non-negative + bounded
        for i in range(n_obs):
            row = np.zeros(n_vars); row[3 + i] = 1.0
            A_rows.append(row)
            l_list.append(0.0)
            u_list.append(self.max_relaxation)

        if not A_rows:
            self.num_solves += 1
            return v_rl.copy(), 0.0

        A = sp.csr_matrix(np.array(A_rows))
        l = np.array(l_list, dtype=float)
        u = np.array([u if np.isfinite(u) else 1e10 for u in u_list], dtype=float)

        # ---- Solve ----
        try:
            prob = osqp.OSQP()
            prob.setup(P, q, A, l, u, **self.solver_settings)
            result = prob.solve()

            if result.info.status not in ('solved', 'solved inaccurate'):
                # pit #6: QP failure -> fallback, no raise
                logger.debug("QP infeasible (status=%s), fallback to v_rl", result.info.status)
                v_safe = v_rl.copy()
                self.num_failures += 1
            else:
                v_safe = result.x[:3]
        except Exception as exc:
            # pit #6: any exception -> fallback
            logger.warning("QP solver exception: %s; falling back to v_rl", exc)
            v_safe = v_rl.copy()
            self.num_failures += 1

        intervention = float(np.linalg.norm(v_safe - v_rl))

        self.num_solves   += 1
        self.total_time   += time.perf_counter() - t0
        self.intervention_history.append(intervention)
        if len(self.intervention_history) > 1000:
            self.intervention_history = self.intervention_history[-1000:]

        return v_safe, intervention

    # ------------------------------------------------------------------

    def get_statistics(self) -> dict:
        if self.num_solves == 0:
            return {'num_solves': 0, 'success_rate': 0.0,
                    'avg_solve_ms': 0.0, 'avg_intervention': 0.0}
        return {
            'num_solves':      self.num_solves,
            'num_failures':    self.num_failures,
            'success_rate':    (self.num_solves - self.num_failures) / self.num_solves,
            'avg_solve_ms':    self.total_time / self.num_solves * 1000,
            'avg_intervention': float(np.mean(self.intervention_history))
                                if self.intervention_history else 0.0,
        }

    def reset_statistics(self):
        self.num_solves = 0
        self.num_failures = 0
        self.total_time = 0.0
        self.intervention_history = []
