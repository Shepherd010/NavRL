"""Unit tests for CurriculumManager."""

import pytest
from omegaconf import OmegaConf

# conftest.py patches Isaac Sim imports; CurriculumManager is pure Python.
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from curriculum_manager import CurriculumManager


def _make_cfg(**overrides):
    """Build a minimal curriculum DictConfig for testing."""
    base = {
        "enabled": True,
        "stage": 1,
        "auto_advance": True,
        "stage1": {
            "num_obstacles": 0,
            "dyn_obstacles": 0,
            "spawn_mode": "center",
            "target_mode": "direction",
            "target_distance": [3.0, 8.0],
            "max_episode_length": 300,
            "speed_target": 2.0,
            "speed_sigma": 1.0,
            "reward_preset": "stable_flight",
        },
        "stage2": {
            "num_obstacles": 0,
            "dyn_obstacles": 0,
            "spawn_mode": "edge",
            "target_mode": "point",
            "target_distance": [10.0, 30.0],
            "max_episode_length": 600,
            "speed_target": 2.5,
            "speed_sigma": 1.0,
            "reward_preset": "goal_navigation",
        },
        "stage3": {
            "num_obstacles": 350,
            "dyn_obstacles": 80,
            "spawn_mode": "edge",
            "target_mode": "point",
            "target_distance": [10.0, 48.0],
            "max_episode_length": 1000,
            "speed_target": 3.0,
            "speed_sigma": 1.0,
            "reward_preset": "full_navigation",
        },
        "advance": {
            "stage1_to_2": {
                "min_episode_len": 200,
                "avg_speed": 1.5,
                "heading_accuracy": 0.6,
                "window": 5,
            },
            "stage2_to_3": {
                "reach_goal_rate": 0.3,
                "avg_speed": 2.0,
                "window": 5,
            },
        },
        "teacher": {
            "enabled": True,
            "checkpoint": "/tmp/fake.pt",
            "weight": 1.0,
            "min_weight": 0.0,
            "warmup_steps": 100,
            "decay_steps": 2000,
            "stages": [2, 3],
        },
        "dagger": {
            "enabled": False,
            "mix_frames": 1e6,
        },
    }
    base.update(overrides)
    return OmegaConf.create(base)


class TestCurriculumManagerInit:
    def test_default_stage(self):
        mgr = CurriculumManager(_make_cfg())
        assert mgr.stage == 1

    def test_custom_start_stage(self):
        mgr = CurriculumManager(_make_cfg(stage=2))
        assert mgr.stage == 2

    def test_get_stage_config_keys(self):
        mgr = CurriculumManager(_make_cfg())
        sc = mgr.get_stage_config()
        for key in ["num_obstacles", "spawn_mode", "target_mode", "reward_preset"]:
            assert key in sc, f"Missing key: {key}"


class TestStageAdvance:
    def _feed(self, mgr, n, ep_len=250, speed=2.0, heading=0.8, reach_goal=0.5):
        for _ in range(n):
            mgr.record_metrics({
                "episode_len": ep_len,
                "speed": speed,
                "heading_accuracy": heading,
                "reach_goal": reach_goal,
            })

    def test_no_advance_insufficient_data(self):
        mgr = CurriculumManager(_make_cfg())
        # Only 1 sample, window is 5, need at least window//2 = 2
        mgr.record_metrics({"episode_len": 250, "speed": 2.0, "heading_accuracy": 0.8})
        assert not mgr.should_advance()

    def test_advance_stage1_to_2(self):
        mgr = CurriculumManager(_make_cfg())
        self._feed(mgr, 5)
        assert mgr.should_advance()

    def test_no_advance_low_speed(self):
        mgr = CurriculumManager(_make_cfg())
        self._feed(mgr, 5, speed=0.5)  # below threshold of 1.5
        assert not mgr.should_advance()

    def test_advance_increments_stage(self):
        mgr = CurriculumManager(_make_cfg())
        self._feed(mgr, 5)
        mgr.advance()
        assert mgr.stage == 2

    def test_no_advance_past_max(self):
        mgr = CurriculumManager(_make_cfg(stage=3))
        self._feed(mgr, 5)
        assert not mgr.should_advance()  # already at max stage

    def test_auto_advance_disabled(self):
        mgr = CurriculumManager(_make_cfg(auto_advance=False))
        self._feed(mgr, 5)
        assert not mgr.should_advance()


class TestTeacherDistill:
    def test_stage1_no_distill(self):
        mgr = CurriculumManager(_make_cfg())
        assert not mgr.teacher_enabled_this_stage()

    def test_stage2_distill(self):
        mgr = CurriculumManager(_make_cfg(stage=2))
        assert mgr.teacher_enabled_this_stage()


class TestStateDictRoundtrip:
    def test_save_load(self):
        mgr = CurriculumManager(_make_cfg())
        for _ in range(5):
            mgr.record_metrics({"episode_len": 250, "speed": 2.0, "heading_accuracy": 0.8})
        mgr.advance()

        state = mgr.state_dict()
        mgr2 = CurriculumManager(_make_cfg())
        mgr2.load_state_dict(state)

        assert mgr2.stage == mgr.stage
        assert mgr2.total_frames == mgr.total_frames
