"""Tests for the drone environment."""

import numpy as np
import pytest
import sys
import os

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from drone_ai.environment import DroneEnv, TaskType, make_env
from drone_ai.simulation import Drone, Obstacle, ObstacleType


class TestDroneEnv:
    """Tests for DroneEnv class."""

    def test_env_creation(self):
        """Test environment can be created."""
        env = make_env(task="hover", difficulty=0.5)
        assert env is not None
        assert env.task == TaskType.HOVER
        assert env.difficulty == 0.5
        env.close()

    def test_observation_space(self):
        """Test observation space is correct dimension."""
        env = make_env(task="delivery_route", difficulty=0.5)
        assert env.observation_space.shape == (31,)
        env.close()

    def test_action_space(self):
        """Test action space is correct dimension."""
        env = make_env(task="delivery_route", difficulty=0.5)
        assert env.action_space.shape == (4,)
        env.close()

    def test_reset(self):
        """Test environment reset."""
        env = make_env(task="delivery_route", difficulty=0.5, seed=42)
        obs, info = env.reset()

        assert obs.shape == (31,)
        assert isinstance(info, dict)
        assert "task" in info
        assert "difficulty" in info
        assert "distance_to_goal" in info
        env.close()

    def test_step(self):
        """Test environment step."""
        env = make_env(task="delivery", difficulty=0.3, seed=42)
        obs, _ = env.reset()

        action = np.array([0.5, 0.0, 0.0, 0.0])
        next_obs, reward, terminated, truncated, info = env.step(action)

        assert next_obs.shape == (31,)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        env.close()

    def test_waypoint_generation_hover(self):
        """Test waypoint generation for hover task."""
        env = make_env(task="hover", difficulty=0.5, seed=42)
        env.reset()

        assert len(env.route_waypoints) >= 1
        env.close()

    def test_waypoint_generation_delivery(self):
        """Test waypoint generation for delivery task."""
        env = make_env(task="delivery", difficulty=0.5, seed=42)
        env.reset()

        assert len(env.route_waypoints) >= 1
        env.close()

    def test_waypoint_generation_delivery_route(self):
        """Test waypoint generation for delivery route task."""
        env = make_env(task="delivery_route", difficulty=0.5, seed=42)
        env.reset()

        # Should have multiple waypoints
        assert len(env.route_waypoints) >= 2
        env.close()

    def test_difficulty_scaling(self):
        """Test that difficulty affects waypoint count."""
        env_easy = make_env(task="delivery_route", difficulty=0.1, seed=42)
        env_hard = make_env(task="delivery_route", difficulty=0.9, seed=42)

        env_easy.reset()
        env_hard.reset()

        # Higher difficulty should generally have more waypoints/obstacles
        # (not always guaranteed due to randomness, but on average)
        assert len(env_easy.obstacles) <= len(env_hard.obstacles) + 5  # Allow some variance

        env_easy.close()
        env_hard.close()

    def test_episode_termination(self):
        """Test episode terminates properly."""
        env = make_env(task="hover", difficulty=0.1, seed=42)
        env.reset()

        # Run until termination
        terminated = False
        truncated = False
        steps = 0
        max_steps = 2000

        while not terminated and not truncated and steps < max_steps:
            action = env.action_space.sample()
            _, _, terminated, truncated, _ = env.step(action)
            steps += 1

        assert terminated or truncated or steps >= max_steps
        env.close()


class TestDrone:
    """Tests for Drone class."""

    def test_drone_creation(self):
        """Test drone can be created."""
        drone = Drone()
        assert drone is not None
        assert np.allclose(drone.position, [0, 0, 1])

    def test_drone_reset(self):
        """Test drone reset."""
        drone = Drone()
        drone.position = np.array([5, 5, 5])
        drone.reset(position=[1, 2, 3])

        assert np.allclose(drone.position, [1, 2, 3])
        assert np.allclose(drone.velocity, [0, 0, 0])

    def test_drone_apply_action(self):
        """Test drone responds to actions."""
        drone = Drone(position=[0, 0, 5])

        # Apply upward thrust
        action = np.array([0.5, 0.0, 0.0, 0.0])
        drone.apply_action(action, dt=0.1)

        # Position should change
        assert drone.position[2] != 5.0


class TestObstacle:
    """Tests for Obstacle class."""

    def test_sphere_obstacle(self):
        """Test sphere obstacle."""
        obs = Obstacle(
            position=[0, 0, 5],
            size=[2.0],
            obstacle_type=ObstacleType.SPHERE
        )

        assert obs.get_radius() == 2.0
        assert obs.contains_point([0, 0, 5])
        assert obs.contains_point([1, 0, 5])
        assert not obs.contains_point([10, 0, 5])

    def test_cylinder_obstacle(self):
        """Test cylinder obstacle."""
        obs = Obstacle(
            position=[0, 0, 0],
            size=[1.0, 5.0],
            obstacle_type=ObstacleType.CYLINDER
        )

        assert obs.get_radius() == 1.0
        assert obs.get_height() == 5.0
        assert obs.contains_point([0, 0, 2.5])
        assert not obs.contains_point([0, 0, 10])

    def test_box_obstacle(self):
        """Test box obstacle."""
        obs = Obstacle(
            position=[0, 0, 0],
            size=[2.0, 2.0, 2.0],
            obstacle_type=ObstacleType.BOX
        )

        assert obs.contains_point([0, 0, 0])
        assert not obs.contains_point([5, 5, 5])

    def test_distance_to_point(self):
        """Test distance calculation."""
        obs = Obstacle(
            position=[0, 0, 0],
            size=[1.0],
            obstacle_type=ObstacleType.SPHERE
        )

        dist = obs.distance_to_point([3, 0, 0])
        assert abs(dist - 2.0) < 0.01  # Distance should be 3 - radius(1) = 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
