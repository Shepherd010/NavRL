"""
conftest.py: patch all Isaac Sim / omni_drones imports so the new pure-PyTorch
modules can be tested without an Isaac installation.

This file is automatically loaded by pytest before any test module.
"""

import sys
from unittest.mock import MagicMock

# All Isaac / omni related module prefixes to mock
_MOCK_PREFIXES = [
    'omni',
    'omni.isaac',
    'omni.isaac.kit',
    'omni.isaac.core',
    'omni.isaac.core.utils',
    'omni.isaac.core.utils.prims',
    'omni.isaac.core.utils.viewports',
    'omni.isaac.orbit',
    'omni.isaac.orbit.sim',
    'omni.isaac.orbit.assets',
    'omni.isaac.orbit.terrains',
    'omni.isaac.orbit.sensors',
    'omni.isaac.orbit.utils',
    'omni.isaac.orbit.utils.math',
    'omni_drones',
    'omni_drones.envs',
    'omni_drones.envs.isaac_env',
    'omni_drones.robots',
    'omni_drones.robots.drone',
    'omni_drones.controllers',
    'omni_drones.utils',
    'omni_drones.utils.torch',
    'omni_drones.utils.torchrl',
    'omni_drones.utils.torchrl.transforms',
    'warp',
]

for name in _MOCK_PREFIXES:
    if name not in sys.modules:
        sys.modules[name] = MagicMock()
