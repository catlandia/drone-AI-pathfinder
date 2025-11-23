"""Tests for path planning algorithms."""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from drone_ai.path_planning import PathPlanner, smooth_path, interpolate_path
from drone_ai.simulation import Obstacle, ObstacleType, generate_random_obstacles


class TestPathPlanner:
    """Tests for PathPlanner class."""

    def test_planner_creation(self):
        """Test planner can be created."""
        obstacles = []
        planner = PathPlanner(obstacles)
        assert planner is not None

    def test_point_validation(self):
        """Test point validation."""
        obstacles = [
            Obstacle(position=[5, 5, 5], size=[2.0], obstacle_type=ObstacleType.SPHERE)
        ]
        planner = PathPlanner(obstacles)

        # Point outside obstacle should be valid
        assert planner.is_point_valid(np.array([0, 0, 5]))

        # Point inside obstacle should be invalid
        assert not planner.is_point_valid(np.array([5, 5, 5]))

        # Point outside bounds should be invalid
        assert not planner.is_point_valid(np.array([100, 0, 5]))

    def test_path_clear(self):
        """Test path clearance check."""
        obstacles = [
            Obstacle(position=[5, 0, 5], size=[2.0], obstacle_type=ObstacleType.SPHERE)
        ]
        planner = PathPlanner(obstacles)

        # Path that doesn't cross obstacle
        assert planner.is_path_clear(np.array([0, 5, 5]), np.array([10, 5, 5]))

        # Path that crosses obstacle
        assert not planner.is_path_clear(np.array([0, 0, 5]), np.array([10, 0, 5]))

    def test_direct_path(self):
        """Test direct path when no obstacles block."""
        obstacles = []
        planner = PathPlanner(obstacles)

        start = np.array([0, 0, 5])
        goal = np.array([10, 10, 5])

        path = planner.plan_path(start, goal)

        assert len(path) == 2
        assert np.allclose(path[0], start)
        assert np.allclose(path[-1], goal)

    def test_astar_path(self):
        """Test A* finds path around obstacle."""
        obstacles = [
            Obstacle(position=[5, 5, 5], size=[3.0], obstacle_type=ObstacleType.SPHERE)
        ]
        planner = PathPlanner(obstacles, grid_resolution=1.0)

        start = np.array([0, 0, 5])
        goal = np.array([10, 10, 5])

        path = planner.plan_path_astar(start, goal)

        # Path should exist
        assert len(path) >= 2

        # Path should start and end correctly
        assert np.linalg.norm(path[0] - start) < 2.0
        assert np.linalg.norm(path[-1] - goal) < 2.0

    def test_rrt_path(self):
        """Test RRT finds path around obstacle."""
        obstacles = [
            Obstacle(position=[5, 5, 5], size=[3.0], obstacle_type=ObstacleType.SPHERE)
        ]
        planner = PathPlanner(obstacles)

        start = np.array([0, 0, 5])
        goal = np.array([10, 10, 5])

        path = planner.plan_path_rrt(start, goal, max_iterations=1000)

        # Path should exist
        assert len(path) >= 2

    def test_avoidance_waypoint(self):
        """Test avoidance waypoint generation."""
        obstacle = Obstacle(position=[5, 0, 5], size=[2.0], obstacle_type=ObstacleType.SPHERE)
        planner = PathPlanner([obstacle])

        current = np.array([0, 0, 5])
        target = np.array([10, 0, 5])

        avoidance = planner.get_avoidance_waypoint(current, target, obstacle)

        # Avoidance point should be away from obstacle
        dist_to_obstacle = np.linalg.norm(avoidance - obstacle.position)
        assert dist_to_obstacle > obstacle.get_radius()


class TestPathUtilities:
    """Tests for path utility functions."""

    def test_smooth_path(self):
        """Test path smoothing."""
        path = [
            np.array([0, 0, 0]),
            np.array([1, 1, 0]),
            np.array([2, 0, 0]),
            np.array([3, 1, 0]),
        ]

        smoothed = smooth_path(path, smoothing_factor=0.5)

        assert len(smoothed) == len(path)
        # Endpoints should be preserved
        assert np.allclose(smoothed[0], path[0])
        assert np.allclose(smoothed[-1], path[-1])

    def test_interpolate_path(self):
        """Test path interpolation."""
        path = [
            np.array([0, 0, 0]),
            np.array([10, 0, 0]),
        ]

        interpolated = interpolate_path(path, max_segment_length=2.0)

        assert len(interpolated) >= 5
        # Check first and last points
        assert np.allclose(interpolated[0], path[0])
        assert np.allclose(interpolated[-1], path[-1])

        # Check segment lengths
        for i in range(len(interpolated) - 1):
            dist = np.linalg.norm(interpolated[i+1] - interpolated[i])
            assert dist <= 2.5  # Allow small tolerance


class TestObstacleGeneration:
    """Tests for obstacle generation."""

    def test_generate_obstacles(self):
        """Test obstacle generation."""
        obstacles = generate_random_obstacles(
            num_obstacles=5,
            bounds=(-10, 10, -10, 10, 0, 10),
            seed=42
        )

        assert len(obstacles) == 5
        for obs in obstacles:
            assert obs.position is not None
            assert obs.size is not None

    def test_obstacle_in_bounds(self):
        """Test generated obstacles are within bounds."""
        bounds = (-10, 10, -10, 10, 0, 10)
        obstacles = generate_random_obstacles(
            num_obstacles=10,
            bounds=bounds,
            seed=42
        )

        for obs in obstacles:
            assert bounds[0] <= obs.position[0] <= bounds[1]
            assert bounds[2] <= obs.position[1] <= bounds[3]
            assert bounds[4] <= obs.position[2] <= bounds[5]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
