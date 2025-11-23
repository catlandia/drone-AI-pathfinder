"""Drone AI - Reinforcement Learning for Autonomous Drone Navigation.

This package implements a curriculum learning approach for training drones
to navigate complex environments with obstacle avoidance and path planning.

Stages: HOVER -> DELIVERY -> DELIVERY_ROUTE
"""

from .simulation import Drone, Obstacle
from .environment import DroneEnv
from .path_planning import PathPlanner
from .visualization import visualize_path

__version__ = "0.1.0"
__all__ = ["Drone", "Obstacle", "DroneEnv", "PathPlanner", "visualize_path"]
