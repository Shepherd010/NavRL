"""Curriculum Manager — stage state-machine for multi-phase training.

Plugin design: instantiate only when ``cfg.curriculum.enabled`` is True.
Provides stage config, auto-advance logic, and checkpoint serialisation.
No Isaac Sim dependency — pure Python + OmegaConf.
"""

from __future__ import annotations
import collections
from typing import Any, Dict, Optional

from omegaconf import DictConfig, OmegaConf


class CurriculumManager:
    """Three-stage curriculum: stable_flight → goal_navigation → full_navigation."""

    STAGE_KEYS = {1: "stage1", 2: "stage2", 3: "stage3"}
    MAX_STAGE = 3

    def __init__(self, cfg: DictConfig):
        """
        Args:
            cfg: the ``curriculum`` sub-config from train.yaml.
        """
        self.cfg = cfg
        self.stage: int = int(OmegaConf.select(cfg, "stage", default=1))
        self.auto_advance: bool = bool(OmegaConf.select(cfg, "auto_advance", default=True))
        self.total_frames: int = 0

        # Rolling metric buffers for auto-advance (keyed by metric name).
        self._metric_buffers: Dict[str, collections.deque] = {}
        self._advance_cfg = OmegaConf.select(cfg, "advance", default=None)

    # ------------------------------------------------------------------
    # Stage accessors
    # ------------------------------------------------------------------

    @property
    def stage_key(self) -> str:
        return self.STAGE_KEYS.get(self.stage, "stage3")

    @property
    def stage_cfg(self) -> DictConfig:
        return self.cfg[self.stage_key]

    @property
    def reward_preset(self) -> str:
        return str(self.stage_cfg.reward_preset)

    @property
    def max_episode_length(self) -> int:
        return int(self.stage_cfg.max_episode_length)

    def get_stage_config(self) -> Dict[str, Any]:
        """Return a plain dict of environment overrides for the current stage."""
        sc = self.stage_cfg
        return {
            "stage": self.stage,
            "num_obstacles": int(sc.num_obstacles),
            "dyn_obstacles": int(sc.dyn_obstacles),
            "spawn_mode": str(sc.spawn_mode),
            "target_mode": str(sc.get("target_mode", "point")),
            "target_distance": list(sc.target_distance),
            "max_episode_length": int(sc.max_episode_length),
            "speed_target": float(sc.speed_target),
            "speed_sigma": float(sc.get("speed_sigma", 1.0)),
            "spawn_radius": float(sc.get("spawn_radius", 3.0)),
            "reward_preset": str(sc.reward_preset),
        }

    # ------------------------------------------------------------------
    # Metric tracking & auto-advance
    # ------------------------------------------------------------------

    def record_metrics(self, metrics: Dict[str, float]) -> None:
        """Push per-episode metrics into rolling buffers."""
        for k, v in metrics.items():
            if k not in self._metric_buffers:
                self._metric_buffers[k] = collections.deque(maxlen=500)
            self._metric_buffers[k].append(v)

    def _buf_mean(self, key: str, window: int) -> Optional[float]:
        buf = self._metric_buffers.get(key)
        if buf is None or len(buf) < max(1, window // 2):
            return None
        recent = list(buf)[-window:]
        return sum(recent) / len(recent)

    def should_advance(self) -> bool:
        """Check if current stage metrics satisfy promotion thresholds.

        Two paths to advance:
        1. Metrics path: rolling-window averages exceed all stage thresholds.
        2. Frame-budget path: total env frames since training start exceed
           ``stage<N>.max_frames`` — acts as a safety net to prevent the policy
           from being stuck in one stage if thresholds are too tight.
        """
        if not self.auto_advance:
            return False
        if self.stage >= self.MAX_STAGE:
            return False

        # --- Frame-budget fallback: force advance if max_frames exhausted ---
        stage_cfg = OmegaConf.select(self.cfg, f"stage{self.stage}", default=None)
        if stage_cfg is not None:
            max_frames = float(stage_cfg.get("max_frames", float("inf")))
            if self.total_frames >= max_frames:
                print(
                    f"[Curriculum] Stage {self.stage} max_frames={max_frames:.0f} reached "
                    f"(total={self.total_frames}) — forcing advance."
                )
                return True

        if self._advance_cfg is None:
            return False

        trans_key = f"stage{self.stage}_to_{self.stage + 1}"
        thresholds = OmegaConf.select(self._advance_cfg, trans_key, default=None)
        if thresholds is None:
            return False

        window = int(thresholds.get("window", 100))

        # Stage 1 → 2
        if self.stage == 1:
            ep_len = self._buf_mean("episode_len", window)
            speed = self._buf_mean("speed", window)
            heading = self._buf_mean("heading_accuracy", window)
            if ep_len is None or speed is None or heading is None:
                return False
            return (
                ep_len >= float(thresholds.min_episode_len)
                and speed >= float(thresholds.avg_speed)
                and heading >= float(thresholds.heading_accuracy)
            )

        # Stage 2 → 3
        if self.stage == 2:
            rg = self._buf_mean("reach_goal", window)
            speed = self._buf_mean("speed", window)
            if rg is None or speed is None:
                return False
            return (
                rg >= float(thresholds.reach_goal_rate)
                and speed >= float(thresholds.avg_speed)
            )

        return False

    def advance(self) -> int:
        """Promote to the next stage. Returns new stage number."""
        if self.stage < self.MAX_STAGE:
            self.stage += 1
            # Clear metric buffers for fresh tracking in the new stage.
            self._metric_buffers.clear()
            print(f"[Curriculum] ★ Advanced to Stage {self.stage}: {self.reward_preset}")
        return self.stage

    # ------------------------------------------------------------------
    # Teacher distillation helpers
    # ------------------------------------------------------------------

    def teacher_enabled_this_stage(self) -> bool:
        teacher_cfg = OmegaConf.select(self.cfg, "teacher", default=None)
        if teacher_cfg is None or not teacher_cfg.get("enabled", False):
            return False
        stages = list(teacher_cfg.get("stages", []))
        return self.stage in stages

    def get_teacher_config(self) -> Optional[DictConfig]:
        teacher_cfg = OmegaConf.select(self.cfg, "teacher", default=None)
        if teacher_cfg is None or not teacher_cfg.get("enabled", False):
            return None
        return teacher_cfg

    # ------------------------------------------------------------------
    # Serialisation (checkpoint resume)
    # ------------------------------------------------------------------

    def state_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "total_frames": self.total_frames,
            "metric_buffers": {k: list(v) for k, v in self._metric_buffers.items()},
        }

    def load_state_dict(self, d: Dict[str, Any]) -> None:
        self.stage = int(d.get("stage", 1))
        self.total_frames = int(d.get("total_frames", 0))
        for k, v in d.get("metric_buffers", {}).items():
            self._metric_buffers[k] = collections.deque(v, maxlen=500)
