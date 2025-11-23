"""Path planning algorithms for drone navigation.

Implements A* and RRT (Rapidly-exploring Random Trees) path planning
with obstacle avoidance for 3D drone navigation.
"""

import numpy as np
from typing import List, Tuple, Optional, Set
from dataclasses import dataclass
import heapq

from .simulation import Obstacle


@dataclass
class Node:
    """Node for path planning algorithms."""
    position: np.ndarray
    parent: Optional['Node'] = None
    g_cost: float = 0.0  # Cost from start
    h_cost: float = 0.0  # Heuristic cost to goal

    @property
    def f_cost(self) -> float:
        """Total cost (g + h)."""
        return self.g_cost + self.h_cost

    def __lt__(self, other: 'Node') -> bool:
        return self.f_cost < other.f_cost

    def __hash__(self) -> int:
        return hash(tuple(np.round(self.position, 2)))

    def __eq__(self, other: 'Node') -> bool:
        if not isinstance(other, Node):
            return False
        return np.allclose(self.position, other.position, atol=0.1)


class PathPlanner:
    """Path planning using A* and RRT algorithms.

    Provides methods for computing obstacle-free paths in 3D space.
    """

    def __init__(
        self,
        obstacles: List[Obstacle],
        grid_resolution: float = 0.5,
        bounds: Tuple[float, float, float, float, float, float] = (-20, 20, -20, 20, 0, 15)
    ):
        """Initialize path planner.

        Args:
            obstacles: List of obstacles in the environment
            grid_resolution: Resolution for A* grid in meters
            bounds: (x_min, x_max, y_min, y_max, z_min, z_max) world bounds
        """
        self.obstacles = obstacles
        self.grid_resolution = grid_resolution
        self.bounds = bounds
        self.safety_margin = 0.5  # Safety margin around obstacles

        # 3D movement directions (26-connected)
        self._directions = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx != 0 or dy != 0 or dz != 0:
                        self._directions.append(np.array([dx, dy, dz]) * grid_resolution)

    def update_obstacles(self, obstacles: List[Obstacle]):
        """Update the obstacle list."""
        self.obstacles = obstacles

    def is_point_valid(self, point: np.ndarray) -> bool:
        """Check if a point is valid (in bounds and not in obstacle).

        Args:
            point: 3D point to check

        Returns:
            True if point is valid
        """
        # Check bounds
        x_min, x_max, y_min, y_max, z_min, z_max = self.bounds
        if not (x_min <= point[0] <= x_max and
                y_min <= point[1] <= y_max and
                z_min <= point[2] <= z_max):
            return False

        # Check obstacle collision
        for obstacle in self.obstacles:
            if obstacle.contains_point(point, self.safety_margin):
                return False

        return True

    def is_path_clear(self, point_a: np.ndarray, point_b: np.ndarray, num_samples: int = 10) -> bool:
        """Check if direct path between two points is obstacle-free.

        Args:
            point_a: Start point
            point_b: End point
            num_samples: Number of points to sample along path

        Returns:
            True if path is clear
        """
        point_a = np.array(point_a)
        point_b = np.array(point_b)

        for t in np.linspace(0, 1, num_samples):
            point = point_a + t * (point_b - point_a)
            if not self.is_point_valid(point):
                return False

        return True

    def _heuristic(self, point: np.ndarray, goal: np.ndarray) -> float:
        """Euclidean distance heuristic."""
        return float(np.linalg.norm(point - goal))

    def plan_path_astar(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        max_iterations: int = 5000
    ) -> List[np.ndarray]:
        """Plan path using A* algorithm.

        Args:
            start: Start position
            goal: Goal position
            max_iterations: Maximum iterations before giving up

        Returns:
            List of waypoints from start to goal, or empty list if no path found
        """
        start = np.array(start, dtype=np.float32)
        goal = np.array(goal, dtype=np.float32)

        # Check if start and goal are valid
        if not self.is_point_valid(start):
            # Try to find nearest valid point
            start = self._find_nearest_valid_point(start)
            if start is None:
                return []

        if not self.is_point_valid(goal):
            goal = self._find_nearest_valid_point(goal)
            if goal is None:
                return []

        # Check direct path first
        if self.is_path_clear(start, goal):
            return [start, goal]

        # A* search
        start_node = Node(start, None, 0, self._heuristic(start, goal))

        open_set: List[Node] = [start_node]
        closed_set: Set[Tuple[float, float, float]] = set()

        iterations = 0
        while open_set and iterations < max_iterations:
            iterations += 1

            # Get node with lowest f_cost
            current = heapq.heappop(open_set)

            # Check if reached goal
            if np.linalg.norm(current.position - goal) < self.grid_resolution * 1.5:
                return self._reconstruct_path(current, goal)

            # Add to closed set
            pos_tuple = tuple(np.round(current.position / self.grid_resolution).astype(int))
            if pos_tuple in closed_set:
                continue
            closed_set.add(pos_tuple)

            # Explore neighbors
            for direction in self._directions:
                neighbor_pos = current.position + direction

                # Skip if invalid
                if not self.is_point_valid(neighbor_pos):
                    continue

                # Skip if already explored
                neighbor_tuple = tuple(np.round(neighbor_pos / self.grid_resolution).astype(int))
                if neighbor_tuple in closed_set:
                    continue

                # Calculate costs
                move_cost = np.linalg.norm(direction)
                g_cost = current.g_cost + move_cost
                h_cost = self._heuristic(neighbor_pos, goal)

                neighbor_node = Node(neighbor_pos, current, g_cost, h_cost)
                heapq.heappush(open_set, neighbor_node)

        # No path found - return direct path anyway
        return [start, goal]

    def _reconstruct_path(self, node: Node, goal: np.ndarray) -> List[np.ndarray]:
        """Reconstruct path from A* result."""
        path = [goal]
        current = node

        while current is not None:
            path.append(current.position)
            current = current.parent

        path.reverse()

        # Simplify path by removing unnecessary waypoints
        return self._simplify_path(path)

    def _simplify_path(self, path: List[np.ndarray]) -> List[np.ndarray]:
        """Remove unnecessary waypoints from path."""
        if len(path) <= 2:
            return path

        simplified = [path[0]]
        i = 0

        while i < len(path) - 1:
            # Find furthest visible point
            furthest = i + 1
            for j in range(i + 2, len(path)):
                if self.is_path_clear(path[i], path[j]):
                    furthest = j

            simplified.append(path[furthest])
            i = furthest

        return simplified

    def _find_nearest_valid_point(
        self,
        point: np.ndarray,
        max_distance: float = 5.0
    ) -> Optional[np.ndarray]:
        """Find nearest valid point to given point."""
        if self.is_point_valid(point):
            return point

        # Search in expanding spheres
        for radius in np.linspace(0.5, max_distance, 10):
            for _ in range(50):
                # Random direction
                direction = np.random.randn(3)
                direction = direction / np.linalg.norm(direction)
                test_point = point + direction * radius

                if self.is_point_valid(test_point):
                    return test_point

        return None

    def plan_path_rrt(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        max_iterations: int = 2000,
        step_size: float = 1.0,
        goal_bias: float = 0.1
    ) -> List[np.ndarray]:
        """Plan path using RRT (Rapidly-exploring Random Trees).

        Args:
            start: Start position
            goal: Goal position
            max_iterations: Maximum iterations
            step_size: Maximum step size for tree extension
            goal_bias: Probability of sampling goal directly

        Returns:
            List of waypoints from start to goal
        """
        start = np.array(start, dtype=np.float32)
        goal = np.array(goal, dtype=np.float32)

        # Validate endpoints
        if not self.is_point_valid(start):
            start = self._find_nearest_valid_point(start)
            if start is None:
                return []

        if not self.is_point_valid(goal):
            goal = self._find_nearest_valid_point(goal)
            if goal is None:
                return []

        # Check direct path
        if self.is_path_clear(start, goal):
            return [start, goal]

        # Initialize tree
        nodes = [Node(start)]

        x_min, x_max, y_min, y_max, z_min, z_max = self.bounds

        for _ in range(max_iterations):
            # Sample random point (with goal bias)
            if np.random.random() < goal_bias:
                sample = goal
            else:
                sample = np.array([
                    np.random.uniform(x_min, x_max),
                    np.random.uniform(y_min, y_max),
                    np.random.uniform(z_min, z_max)
                ])

            # Find nearest node
            distances = [np.linalg.norm(n.position - sample) for n in nodes]
            nearest_idx = np.argmin(distances)
            nearest = nodes[nearest_idx]

            # Extend toward sample
            direction = sample - nearest.position
            distance = np.linalg.norm(direction)

            if distance < 1e-6:
                continue

            direction = direction / distance
            new_distance = min(distance, step_size)
            new_pos = nearest.position + direction * new_distance

            # Check if new position is valid
            if not self.is_path_clear(nearest.position, new_pos):
                continue

            # Add new node
            new_node = Node(new_pos, nearest)
            nodes.append(new_node)

            # Check if goal reached
            if np.linalg.norm(new_pos - goal) < step_size:
                if self.is_path_clear(new_pos, goal):
                    goal_node = Node(goal, new_node)
                    return self._reconstruct_rrt_path(goal_node)

        # Return best partial path
        distances_to_goal = [np.linalg.norm(n.position - goal) for n in nodes]
        best_idx = np.argmin(distances_to_goal)
        return self._reconstruct_rrt_path(nodes[best_idx])

    def _reconstruct_rrt_path(self, node: Node) -> List[np.ndarray]:
        """Reconstruct path from RRT tree."""
        path = []
        current = node

        while current is not None:
            path.append(current.position)
            current = current.parent

        path.reverse()
        return self._simplify_path(path)

    def plan_path(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        algorithm: str = "astar"
    ) -> List[np.ndarray]:
        """Plan path from start to goal avoiding obstacles.

        Args:
            start: Start position
            goal: Goal position
            algorithm: "astar" or "rrt"

        Returns:
            List of waypoints from start to goal
        """
        if algorithm == "rrt":
            return self.plan_path_rrt(start, goal)
        else:
            return self.plan_path_astar(start, goal)

    def get_avoidance_waypoint(
        self,
        current_pos: np.ndarray,
        target_pos: np.ndarray,
        obstacle: Obstacle
    ) -> np.ndarray:
        """Generate a waypoint to avoid a specific obstacle.

        Args:
            current_pos: Current drone position
            target_pos: Target position
            obstacle: Obstacle to avoid

        Returns:
            Waypoint that routes around the obstacle
        """
        current_pos = np.array(current_pos)
        target_pos = np.array(target_pos)

        # Vector from current to target
        to_target = target_pos - current_pos
        to_target_norm = to_target / (np.linalg.norm(to_target) + 1e-8)

        # Vector from current to obstacle
        to_obstacle = obstacle.position - current_pos

        # Calculate perpendicular direction
        # Use cross product with up vector to get horizontal avoidance
        up = np.array([0, 0, 1])
        perp = np.cross(to_target_norm, up)
        perp_norm = np.linalg.norm(perp)

        if perp_norm < 1e-6:
            # Target is directly above/below - use arbitrary perpendicular
            perp = np.array([1, 0, 0])
        else:
            perp = perp / perp_norm

        # Determine which side to go around
        side = np.sign(np.dot(perp, to_obstacle))
        if side == 0:
            side = 1

        # Calculate avoidance point
        obstacle_radius = obstacle.get_radius() + self.safety_margin * 2

        # Point to the side of obstacle
        avoidance_point = obstacle.position - side * perp * obstacle_radius

        # Also consider going above
        above_point = obstacle.position.copy()
        above_point[2] = obstacle.position[2] + obstacle.get_height() + self.safety_margin * 2

        # Choose the closer avoidance option
        dist_side = np.linalg.norm(avoidance_point - current_pos) + np.linalg.norm(avoidance_point - target_pos)
        dist_above = np.linalg.norm(above_point - current_pos) + np.linalg.norm(above_point - target_pos)

        if dist_above < dist_side and self.is_point_valid(above_point):
            return above_point
        elif self.is_point_valid(avoidance_point):
            return avoidance_point
        else:
            # Try the other side
            avoidance_point = obstacle.position + side * perp * obstacle_radius
            if self.is_point_valid(avoidance_point):
                return avoidance_point
            return above_point


def smooth_path(path: List[np.ndarray], smoothing_factor: float = 0.5) -> List[np.ndarray]:
    """Apply smoothing to a path.

    Args:
        path: List of waypoints
        smoothing_factor: How much to smooth (0 = no smoothing, 1 = max smoothing)

    Returns:
        Smoothed path
    """
    if len(path) <= 2:
        return path

    smoothed = [np.array(p) for p in path]

    # Iterative smoothing
    for _ in range(10):
        for i in range(1, len(smoothed) - 1):
            prev_pos = smoothed[i].copy()

            # Move toward midpoint of neighbors
            midpoint = (smoothed[i-1] + smoothed[i+1]) / 2
            smoothed[i] = smoothed[i] + smoothing_factor * (midpoint - smoothed[i])

    return smoothed


def interpolate_path(path: List[np.ndarray], max_segment_length: float = 1.0) -> List[np.ndarray]:
    """Interpolate path to have smaller segment lengths.

    Args:
        path: List of waypoints
        max_segment_length: Maximum distance between consecutive points

    Returns:
        Interpolated path with more waypoints
    """
    if len(path) < 2:
        return path

    interpolated = [path[0]]

    for i in range(1, len(path)):
        start = path[i-1]
        end = path[i]

        distance = np.linalg.norm(end - start)
        num_segments = max(1, int(np.ceil(distance / max_segment_length)))

        for j in range(1, num_segments + 1):
            t = j / num_segments
            point = start + t * (end - start)
            interpolated.append(point)

    return interpolated
