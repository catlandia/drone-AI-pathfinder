"""Drone navigation environment with curriculum learning.

Implements a Gymnasium-compatible reinforcement learning environment
for training drones with path selection and obstacle avoidance.

Stages: HOVER -> DELIVERY -> DELIVERY_ROUTE
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
from dataclasses import dataclass
import gymnasium as gym
from gymnasium import spaces

from .simulation import Drone, Obstacle, ObstacleType, generate_random_obstacles
from .path_planning import PathPlanner


class TaskType(Enum):
    """Training task types in curriculum order."""
    HOVER = "hover"
    DELIVERY = "delivery"
    DELIVERY_ROUTE = "delivery_route"


class RouteShape(Enum):
    """Types of route shapes for waypoint generation."""
    STRAIGHT = "straight"
    CURVED = "curved"
    S_CURVE = "s_curve"
    ZIGZAG = "zigzag"
    TERRAIN_FOLLOWING = "terrain_following"
    SPIRAL = "spiral"


@dataclass
class EnvironmentConfig:
    """Configuration for the drone environment."""
    # World bounds
    x_bounds: Tuple[float, float] = (-25.0, 25.0)
    y_bounds: Tuple[float, float] = (-25.0, 25.0)
    z_bounds: Tuple[float, float] = (0.0, 15.0)

    # Simulation
    dt: float = 0.02  # 50 Hz simulation
    max_episode_steps: int = 1000

    # Obstacles
    num_obstacles: int = 8
    obstacle_min_size: float = 0.5
    obstacle_max_size: float = 2.0

    # Waypoints
    max_waypoints: int = 6
    waypoint_reach_distance: float = 1.0

    # Safety
    drone_radius: float = 0.3
    min_altitude: float = 0.5
    max_altitude: float = 12.0


class DroneEnv(gym.Env):
    """Gymnasium environment for drone navigation with path selection.

    Observation Space (31 dimensions):
        [0:3]   - Drone position (x, y, z)
        [3:6]   - Drone velocity (vx, vy, vz)
        [6:9]   - Drone orientation (roll, pitch, yaw)
        [9:12]  - To current waypoint vector (dx, dy, dz)
        [12]    - Distance to current waypoint
        [13:16] - To next waypoint vector (if exists)
        [16]    - Distance to next waypoint
        [17:20] - To waypoint after next (if exists)
        [20]    - Bearing angle to next waypoint
        [21]    - Path curvature/turning angle ahead
        [22]    - Remaining waypoints count (normalized)
        [23]    - Nearest obstacle distance
        [24:27] - Nearest obstacle direction
        [27]    - Obstacle proximity in travel direction
        [28:31] - Goal position

    Action Space (4 dimensions):
        [0] - Thrust command [-1, 1]
        [1] - Roll rate [-1, 1]
        [2] - Pitch rate [-1, 1]
        [3] - Yaw rate [-1, 1]
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        task: TaskType = TaskType.HOVER,
        difficulty: float = 0.5,
        config: EnvironmentConfig = None,
        render_mode: str = None,
        seed: int = None
    ):
        """Initialize drone environment.

        Args:
            task: Current training task
            difficulty: Difficulty level 0.0 (easy) to 1.0 (hard)
            config: Environment configuration
            render_mode: Rendering mode ("human" or "rgb_array")
            seed: Random seed
        """
        super().__init__()

        self.task = task
        self.difficulty = np.clip(difficulty, 0.0, 1.0)
        self.config = config or EnvironmentConfig()
        self.render_mode = render_mode

        if seed is not None:
            np.random.seed(seed)

        # Initialize components
        self.drone = Drone()
        self.obstacles: List[Obstacle] = []
        self.path_planner: Optional[PathPlanner] = None

        # Waypoint tracking
        self.route_waypoints: List[np.ndarray] = []
        self.current_waypoint_idx: int = 0
        self.goal_position: np.ndarray = np.zeros(3)

        # Episode tracking
        self.step_count: int = 0
        self.total_distance_traveled: float = 0.0
        self.previous_position: np.ndarray = np.zeros(3)
        self.previous_heading: float = 0.0
        self.heading_changes: List[float] = []

        # Reward weights - tuned for path following
        self.reward_weights = {
            # Base rewards
            "waypoint_reached": 10.0,
            "goal_reached": 50.0,
            "collision": -50.0,
            "out_of_bounds": -30.0,
            "crash": -100.0,

            # Distance rewards
            "distance_to_waypoint": -0.1,  # Per meter
            "progress_toward_waypoint": 2.0,

            # Path efficiency rewards
            "path_efficiency": 1.0,  # Bonus for shorter paths
            "smooth_path": -0.5,  # Penalty per radian of heading change
            "altitude_band": 0.5,  # Reward for optimal altitude
            "lookahead_alignment": 0.3,  # Reward for heading toward future waypoints

            # Safety rewards
            "obstacle_proximity": -1.0,  # Penalty when close to obstacles
            "altitude_penalty": -0.5,  # Penalty for extreme altitudes

            # Time penalty
            "time_step": -0.01,
        }

        # Define spaces
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(31,),
            dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32
        )

        # Visualization
        self._renderer = None

    def reset(
        self,
        seed: int = None,
        options: Dict = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state.

        Args:
            seed: Random seed
            options: Additional options (can include 'difficulty', 'task')

        Returns:
            Initial observation and info dict
        """
        super().reset(seed=seed)

        if seed is not None:
            np.random.seed(seed)

        # Apply options
        if options:
            if 'difficulty' in options:
                self.difficulty = np.clip(options['difficulty'], 0.0, 1.0)
            if 'task' in options:
                self.task = TaskType(options['task'])

        # Reset tracking
        self.step_count = 0
        self.total_distance_traveled = 0.0
        self.heading_changes = []

        # Generate environment based on task
        self._setup_environment()

        # Initialize path planner
        bounds = (
            self.config.x_bounds[0], self.config.x_bounds[1],
            self.config.y_bounds[0], self.config.y_bounds[1],
            self.config.z_bounds[0], self.config.z_bounds[1]
        )
        self.path_planner = PathPlanner(self.obstacles, bounds=bounds)

        # Track position for distance calculation
        self.previous_position = self.drone.position.copy()
        self.previous_heading = self.drone.orientation[2]

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def _setup_environment(self):
        """Setup environment based on current task and difficulty."""
        if self.task == TaskType.HOVER:
            self._setup_hover_task()
        elif self.task == TaskType.DELIVERY:
            self._setup_delivery_task()
        else:  # DELIVERY_ROUTE
            self._setup_delivery_route_task()

    def _setup_hover_task(self):
        """Setup simple hover task."""
        # Start at random position
        start_pos = np.array([
            np.random.uniform(-5, 5),
            np.random.uniform(-5, 5),
            np.random.uniform(2, 5)
        ])
        self.drone.reset(start_pos)

        # Goal is to hover at a nearby point
        self.goal_position = start_pos + np.random.uniform(-2, 2, 3)
        self.goal_position[2] = np.clip(self.goal_position[2], 1, 10)

        self.route_waypoints = [self.goal_position]
        self.current_waypoint_idx = 0

        # Few or no obstacles for hover
        num_obs = int(self.difficulty * 3)
        self.obstacles = generate_random_obstacles(
            num_obs,
            (self.config.x_bounds[0], self.config.x_bounds[1],
             self.config.y_bounds[0], self.config.y_bounds[1],
             0.5, 8),
            seed=np.random.randint(10000)
        )

    def _setup_delivery_task(self):
        """Setup point-to-point delivery task."""
        # Random start position
        start_pos = np.array([
            np.random.uniform(-10, 10),
            np.random.uniform(-10, 10),
            np.random.uniform(2, 6)
        ])
        self.drone.reset(start_pos)

        # Goal at distance based on difficulty
        distance = 10 + self.difficulty * 20
        angle = np.random.uniform(0, 2 * np.pi)
        self.goal_position = start_pos + np.array([
            np.cos(angle) * distance,
            np.sin(angle) * distance,
            np.random.uniform(-2, 2)
        ])
        self.goal_position = np.clip(
            self.goal_position,
            [self.config.x_bounds[0] + 2, self.config.y_bounds[0] + 2, 1],
            [self.config.x_bounds[1] - 2, self.config.y_bounds[1] - 2, 12]
        )

        self.route_waypoints = [self.goal_position]
        self.current_waypoint_idx = 0

        # Obstacles based on difficulty
        num_obs = int(3 + self.difficulty * 8)
        self.obstacles = generate_random_obstacles(
            num_obs,
            (self.config.x_bounds[0], self.config.x_bounds[1],
             self.config.y_bounds[0], self.config.y_bounds[1],
             0.5, 10),
            seed=np.random.randint(10000)
        )

    def _setup_delivery_route_task(self):
        """Setup multi-waypoint delivery route task."""
        # Random start position
        start_pos = np.array([
            np.random.uniform(-15, -5),
            np.random.uniform(-5, 5),
            np.random.uniform(2, 5)
        ])
        self.drone.reset(start_pos)

        # Generate obstacles first (so waypoints can avoid them)
        num_obs = int(5 + self.difficulty * 10)
        self.obstacles = generate_random_obstacles(
            num_obs,
            (self.config.x_bounds[0] + 3, self.config.x_bounds[1] - 3,
             self.config.y_bounds[0] + 3, self.config.y_bounds[1] - 3,
             0.5, 10),
            min_size=self.config.obstacle_min_size,
            max_size=self.config.obstacle_max_size,
            seed=np.random.randint(10000)
        )

        # Generate route waypoints
        self.route_waypoints = self._generate_route_waypoints(start_pos)
        self.current_waypoint_idx = 0
        self.goal_position = self.route_waypoints[-1] if self.route_waypoints else start_pos

    def _generate_route_waypoints(self, start: np.ndarray) -> List[np.ndarray]:
        """Generate waypoints for DELIVERY_ROUTE task with varied shapes.

        Creates diverse route patterns based on difficulty:
        - Low difficulty: Simple curved paths, 2-3 waypoints
        - Medium difficulty: S-curves, zigzags, 3-4 waypoints
        - High difficulty: Complex terrain-following, spirals, 4-6 waypoints

        Args:
            start: Starting position

        Returns:
            List of waypoint positions
        """
        # Number of waypoints scales with difficulty
        num_waypoints = int(2 + self.difficulty * 4)
        num_waypoints = min(num_waypoints, self.config.max_waypoints)

        # Select route shape based on difficulty
        if self.difficulty < 0.3:
            shapes = [RouteShape.STRAIGHT, RouteShape.CURVED]
        elif self.difficulty < 0.6:
            shapes = [RouteShape.CURVED, RouteShape.S_CURVE, RouteShape.ZIGZAG]
        else:
            shapes = [RouteShape.S_CURVE, RouteShape.ZIGZAG, RouteShape.TERRAIN_FOLLOWING, RouteShape.SPIRAL]

        shape = np.random.choice(shapes)

        # Generate goal position
        distance = 15 + self.difficulty * 20
        base_angle = np.random.uniform(0, 2 * np.pi)
        goal = start + np.array([
            np.cos(base_angle) * distance,
            np.sin(base_angle) * distance,
            np.random.uniform(-1, 3)
        ])
        goal = self._clamp_to_bounds(goal)

        # Generate waypoints based on shape
        if shape == RouteShape.STRAIGHT:
            waypoints = self._generate_straight_route(start, goal, num_waypoints)
        elif shape == RouteShape.CURVED:
            waypoints = self._generate_curved_route(start, goal, num_waypoints)
        elif shape == RouteShape.S_CURVE:
            waypoints = self._generate_s_curve_route(start, goal, num_waypoints)
        elif shape == RouteShape.ZIGZAG:
            waypoints = self._generate_zigzag_route(start, goal, num_waypoints)
        elif shape == RouteShape.TERRAIN_FOLLOWING:
            waypoints = self._generate_terrain_following_route(start, goal, num_waypoints)
        else:  # SPIRAL
            waypoints = self._generate_spiral_route(start, goal, num_waypoints)

        # Adjust waypoints to avoid obstacles
        waypoints = self._adjust_waypoints_for_obstacles(waypoints)

        # Add altitude variation based on difficulty
        waypoints = self._add_altitude_variation(waypoints)

        return waypoints

    def _generate_straight_route(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        num_waypoints: int
    ) -> List[np.ndarray]:
        """Generate simple straight-line route with intermediate waypoints."""
        waypoints = []
        for i in range(1, num_waypoints + 1):
            t = i / num_waypoints
            point = start + t * (goal - start)
            # Add small random offset
            offset = np.random.uniform(-1, 1, 3) * (1 - t) * self.difficulty
            offset[2] *= 0.5  # Less vertical variation
            waypoints.append(self._clamp_to_bounds(point + offset))
        return waypoints

    def _generate_curved_route(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        num_waypoints: int
    ) -> List[np.ndarray]:
        """Generate curved/arc route."""
        waypoints = []

        # Calculate perpendicular direction for curve
        direction = goal - start
        direction_norm = direction / (np.linalg.norm(direction) + 1e-8)

        # Perpendicular in XY plane
        perp = np.array([-direction_norm[1], direction_norm[0], 0])
        curve_magnitude = np.linalg.norm(direction) * 0.3 * (0.5 + self.difficulty)

        # Random curve direction
        curve_sign = np.random.choice([-1, 1])

        for i in range(1, num_waypoints + 1):
            t = i / num_waypoints
            # Base position along line
            base = start + t * direction

            # Add curved offset (parabolic)
            curve_offset = curve_sign * perp * curve_magnitude * 4 * t * (1 - t)

            waypoints.append(self._clamp_to_bounds(base + curve_offset))

        return waypoints

    def _generate_s_curve_route(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        num_waypoints: int
    ) -> List[np.ndarray]:
        """Generate S-curve route with two turns."""
        waypoints = []

        direction = goal - start
        direction_norm = direction / (np.linalg.norm(direction) + 1e-8)
        perp = np.array([-direction_norm[1], direction_norm[0], 0])

        curve_magnitude = np.linalg.norm(direction) * 0.25 * (0.5 + self.difficulty)

        for i in range(1, num_waypoints + 1):
            t = i / num_waypoints
            base = start + t * direction

            # S-curve using sine wave
            s_offset = perp * curve_magnitude * np.sin(t * 2 * np.pi)

            waypoints.append(self._clamp_to_bounds(base + s_offset))

        return waypoints

    def _generate_zigzag_route(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        num_waypoints: int
    ) -> List[np.ndarray]:
        """Generate zigzag route with sharp turns."""
        waypoints = []

        direction = goal - start
        direction_norm = direction / (np.linalg.norm(direction) + 1e-8)
        perp = np.array([-direction_norm[1], direction_norm[0], 0])

        zigzag_magnitude = 3 + self.difficulty * 5

        for i in range(1, num_waypoints + 1):
            t = i / num_waypoints
            base = start + t * direction

            # Alternating zigzag
            sign = 1 if i % 2 == 0 else -1
            zigzag_offset = perp * zigzag_magnitude * sign

            waypoints.append(self._clamp_to_bounds(base + zigzag_offset))

        return waypoints

    def _generate_terrain_following_route(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        num_waypoints: int
    ) -> List[np.ndarray]:
        """Generate terrain-following route with altitude changes."""
        waypoints = []

        direction = goal - start

        for i in range(1, num_waypoints + 1):
            t = i / num_waypoints
            base = start + t * direction

            # Simulated terrain with varying altitude
            terrain_height = 2 + 3 * np.sin(t * 3 * np.pi) * self.difficulty
            terrain_height += np.random.uniform(-1, 1)

            point = base.copy()
            point[2] = np.clip(terrain_height, self.config.min_altitude, self.config.max_altitude)

            waypoints.append(self._clamp_to_bounds(point))

        return waypoints

    def _generate_spiral_route(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        num_waypoints: int
    ) -> List[np.ndarray]:
        """Generate spiral/helical route."""
        waypoints = []

        center = (start + goal) / 2
        radius = np.linalg.norm(goal - start) * 0.3
        start_angle = np.arctan2(start[1] - center[1], start[0] - center[0])

        # Spiral parameters
        turns = 0.5 + self.difficulty
        height_change = goal[2] - start[2]

        for i in range(1, num_waypoints + 1):
            t = i / num_waypoints
            angle = start_angle + t * turns * 2 * np.pi

            # Decreasing radius spiral
            current_radius = radius * (1 - t * 0.5)

            point = np.array([
                center[0] + current_radius * np.cos(angle),
                center[1] + current_radius * np.sin(angle),
                start[2] + t * height_change
            ])

            waypoints.append(self._clamp_to_bounds(point))

        # Ensure last waypoint is the goal
        waypoints[-1] = goal

        return waypoints

    def _adjust_waypoints_for_obstacles(
        self,
        waypoints: List[np.ndarray]
    ) -> List[np.ndarray]:
        """Adjust waypoints to avoid obstacles."""
        adjusted = []

        for waypoint in waypoints:
            # Check if waypoint is inside any obstacle
            in_obstacle = False
            for obstacle in self.obstacles:
                if obstacle.contains_point(waypoint, margin=1.5):
                    in_obstacle = True
                    # Move waypoint away from obstacle
                    direction = waypoint - obstacle.position
                    if np.linalg.norm(direction) < 0.1:
                        direction = np.array([1, 0, 0])
                    direction = direction / np.linalg.norm(direction)

                    safe_distance = obstacle.get_radius() + 2.0
                    new_waypoint = obstacle.position + direction * safe_distance
                    waypoint = self._clamp_to_bounds(new_waypoint)
                    break

            adjusted.append(waypoint)

        return adjusted

    def _add_altitude_variation(self, waypoints: List[np.ndarray]) -> List[np.ndarray]:
        """Add altitude variation to waypoints based on difficulty."""
        if self.difficulty < 0.3:
            return waypoints

        varied = []
        for i, wp in enumerate(waypoints):
            # Add some altitude variation
            altitude_offset = np.sin(i * np.pi / 2) * 2 * self.difficulty
            new_wp = wp.copy()
            new_wp[2] = np.clip(
                wp[2] + altitude_offset,
                self.config.min_altitude,
                self.config.max_altitude
            )
            varied.append(new_wp)

        return varied

    def _clamp_to_bounds(self, point: np.ndarray) -> np.ndarray:
        """Clamp point to world bounds."""
        return np.clip(
            point,
            [self.config.x_bounds[0] + 1, self.config.y_bounds[0] + 1, self.config.z_bounds[0] + 0.5],
            [self.config.x_bounds[1] - 1, self.config.y_bounds[1] - 1, self.config.z_bounds[1] - 1]
        )

    def _get_observation(self) -> np.ndarray:
        """Build 31-dimensional observation vector.

        Includes path selection features:
        - Multi-waypoint lookahead (next 3 waypoints)
        - Bearing angles
        - Path curvature
        - Obstacle proximity in travel direction

        Returns:
            31-dimensional numpy array
        """
        obs = np.zeros(31, dtype=np.float32)

        # [0:3] Drone position
        obs[0:3] = self.drone.position

        # [3:6] Drone velocity
        obs[3:6] = self.drone.velocity

        # [6:9] Drone orientation
        obs[6:9] = self.drone.orientation

        # Current waypoint info
        current_wp = self._get_current_waypoint()

        # [9:12] Vector to current waypoint
        to_current = current_wp - self.drone.position
        obs[9:12] = to_current

        # [12] Distance to current waypoint
        obs[12] = np.linalg.norm(to_current)

        # [13:16] Vector to next waypoint
        next_wp = self._get_waypoint(self.current_waypoint_idx + 1)
        if next_wp is not None:
            to_next = next_wp - self.drone.position
            obs[13:16] = to_next
            obs[16] = np.linalg.norm(to_next)
        else:
            obs[13:16] = to_current
            obs[16] = obs[12]

        # [17:20] Vector to waypoint after next
        wp_after = self._get_waypoint(self.current_waypoint_idx + 2)
        if wp_after is not None:
            obs[17:20] = wp_after - self.drone.position

        # [20] Bearing angle to next waypoint
        obs[20] = self._calculate_bearing_angle(to_current)

        # [21] Path curvature/turning angle ahead
        obs[21] = self._calculate_path_curvature()

        # [22] Remaining waypoints (normalized)
        remaining = len(self.route_waypoints) - self.current_waypoint_idx
        obs[22] = remaining / max(len(self.route_waypoints), 1)

        # [23] Nearest obstacle distance
        nearest_dist, nearest_obs = self.drone.get_nearest_obstacle_distance(self.obstacles)
        obs[23] = min(nearest_dist, 20.0) / 20.0  # Normalized

        # [24:27] Direction to nearest obstacle
        if nearest_obs is not None:
            to_obs = nearest_obs.position - self.drone.position
            obs[24:27] = to_obs / (np.linalg.norm(to_obs) + 1e-8)

        # [27] Obstacle proximity in travel direction
        obs[27] = self._get_obstacle_proximity_ahead()

        # [28:31] Goal position
        obs[28:31] = self.goal_position

        return obs

    def _get_current_waypoint(self) -> np.ndarray:
        """Get current target waypoint."""
        if self.current_waypoint_idx < len(self.route_waypoints):
            return self.route_waypoints[self.current_waypoint_idx]
        return self.goal_position

    def _get_waypoint(self, index: int) -> Optional[np.ndarray]:
        """Get waypoint at specific index, or None if out of range."""
        if 0 <= index < len(self.route_waypoints):
            return self.route_waypoints[index]
        return None

    def _calculate_bearing_angle(self, to_waypoint: np.ndarray) -> float:
        """Calculate bearing angle between drone heading and waypoint direction."""
        if np.linalg.norm(to_waypoint[:2]) < 1e-6:
            return 0.0

        # Current heading from yaw
        heading = self.drone.orientation[2]
        heading_vec = np.array([np.cos(heading), np.sin(heading)])

        # Direction to waypoint in XY plane
        wp_direction = to_waypoint[:2] / (np.linalg.norm(to_waypoint[:2]) + 1e-8)

        # Angle between them
        dot = np.clip(np.dot(heading_vec, wp_direction), -1, 1)
        angle = np.arccos(dot)

        # Determine sign (left or right)
        cross = heading_vec[0] * wp_direction[1] - heading_vec[1] * wp_direction[0]
        return angle * np.sign(cross)

    def _calculate_path_curvature(self) -> float:
        """Calculate path curvature at current position."""
        current_wp = self._get_current_waypoint()
        next_wp = self._get_waypoint(self.current_waypoint_idx + 1)

        if next_wp is None:
            return 0.0

        # Vectors
        to_current = current_wp - self.drone.position
        current_to_next = next_wp - current_wp

        # Normalize
        to_current_norm = to_current / (np.linalg.norm(to_current) + 1e-8)
        to_next_norm = current_to_next / (np.linalg.norm(current_to_next) + 1e-8)

        # Angle between directions
        dot = np.clip(np.dot(to_current_norm, to_next_norm), -1, 1)
        return np.arccos(dot)

    def _get_obstacle_proximity_ahead(self) -> float:
        """Get closest obstacle distance in travel direction."""
        # Travel direction based on velocity or waypoint direction
        if np.linalg.norm(self.drone.velocity) > 0.5:
            direction = self.drone.velocity
        else:
            direction = self._get_current_waypoint() - self.drone.position

        obstacles_ahead = self.drone.get_obstacle_distances_in_direction(
            self.obstacles, direction, cone_angle=np.pi/3
        )

        if obstacles_ahead:
            return min(obstacles_ahead[0][0], 20.0) / 20.0
        return 1.0  # No obstacles ahead

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one environment step.

        Args:
            action: 4D action vector [thrust, roll_rate, pitch_rate, yaw_rate]

        Returns:
            observation, reward, terminated, truncated, info
        """
        self.step_count += 1

        # Apply action
        self.drone.apply_action(action, self.config.dt)

        # Update obstacles (for moving obstacles)
        for obstacle in self.obstacles:
            obstacle.update(self.config.dt)

        # Check for waypoint reached
        self._check_waypoint_reached()

        # Check for path replanning need
        self._check_replan_needed()

        # Update tracking
        distance = np.linalg.norm(self.drone.position - self.previous_position)
        self.total_distance_traveled += distance

        heading_change = abs(self.drone.orientation[2] - self.previous_heading)
        if heading_change > np.pi:
            heading_change = 2 * np.pi - heading_change
        self.heading_changes.append(heading_change)

        self.previous_position = self.drone.position.copy()
        self.previous_heading = self.drone.orientation[2]

        # Compute reward
        reward = self._compute_reward()

        # Check termination conditions
        terminated, truncated = self._check_termination()

        # Get observation
        observation = self._get_observation()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _check_waypoint_reached(self):
        """Check if current waypoint has been reached."""
        current_wp = self._get_current_waypoint()
        distance = np.linalg.norm(self.drone.position - current_wp)

        if distance < self.config.waypoint_reach_distance:
            if self.current_waypoint_idx < len(self.route_waypoints) - 1:
                self.current_waypoint_idx += 1

    def _check_replan_needed(self):
        """Check if path replanning is needed and trigger if so."""
        # Check for obstacle in direct path to next waypoint
        current_wp = self._get_current_waypoint()
        to_waypoint = current_wp - self.drone.position
        distance_to_wp = np.linalg.norm(to_waypoint)

        if distance_to_wp < 0.5:
            return  # Too close, don't replan

        # Check obstacles in path
        direction = to_waypoint / distance_to_wp
        obstacles_ahead = self.drone.get_obstacle_distances_in_direction(
            self.obstacles, direction, cone_angle=np.pi/6
        )

        # Trigger replan if obstacle is close and blocking
        for dist, obstacle in obstacles_ahead:
            if dist < 3.0 and dist < distance_to_wp:
                self._replan_path(obstacle)
                break

    def _replan_path(self, blocking_obstacle: Obstacle):
        """Dynamically replan path around obstacle.

        Generates alternative waypoints around the blocking obstacle
        while maintaining overall direction toward goal.

        Args:
            blocking_obstacle: The obstacle blocking the current path
        """
        if self.path_planner is None:
            return

        current_wp = self._get_current_waypoint()

        # Generate avoidance waypoint
        avoidance_wp = self.path_planner.get_avoidance_waypoint(
            self.drone.position,
            current_wp,
            blocking_obstacle
        )

        # Insert avoidance waypoint before current target
        if self.current_waypoint_idx < len(self.route_waypoints):
            # Check if avoidance point is valid and helpful
            dist_to_avoidance = np.linalg.norm(avoidance_wp - self.drone.position)
            dist_to_current = np.linalg.norm(current_wp - self.drone.position)

            # Only insert if it's actually helping (not too far out of way)
            if dist_to_avoidance < dist_to_current * 1.5:
                self.route_waypoints.insert(self.current_waypoint_idx, avoidance_wp)

    def _compute_reward(self) -> float:
        """Compute reward for current state.

        Includes path-following rewards:
        - path_efficiency: Bonus for efficient paths
        - smooth_path: Penalty for heading changes
        - altitude_band: Reward for staying in optimal altitude
        - lookahead_alignment: Reward for heading toward future waypoints

        Returns:
            Total reward value
        """
        reward = 0.0

        # Time penalty
        reward += self.reward_weights["time_step"]

        # Check collision
        if self.drone.check_collision(self.obstacles, self.config.drone_radius):
            return self.reward_weights["collision"]

        if self.drone.crashed:
            return self.reward_weights["crash"]

        # Check bounds
        if not self._in_bounds():
            return self.reward_weights["out_of_bounds"]

        # Distance to current waypoint
        current_wp = self._get_current_waypoint()
        distance = np.linalg.norm(self.drone.position - current_wp)
        reward += self.reward_weights["distance_to_waypoint"] * distance

        # Progress toward waypoint
        prev_distance = np.linalg.norm(self.previous_position - current_wp)
        progress = prev_distance - distance
        reward += self.reward_weights["progress_toward_waypoint"] * progress

        # Waypoint reached bonus
        if distance < self.config.waypoint_reach_distance:
            if self.current_waypoint_idx >= len(self.route_waypoints) - 1:
                reward += self.reward_weights["goal_reached"]
            else:
                reward += self.reward_weights["waypoint_reached"]

        # Path efficiency reward
        if self.total_distance_traveled > 0:
            direct_distance = np.linalg.norm(self.goal_position - self.drone.position)
            efficiency = direct_distance / (self.total_distance_traveled + direct_distance + 1e-8)
            reward += self.reward_weights["path_efficiency"] * efficiency

        # Smooth path reward (penalty for heading changes)
        if self.heading_changes:
            recent_changes = self.heading_changes[-10:]
            avg_change = np.mean(recent_changes)
            reward += self.reward_weights["smooth_path"] * avg_change

        # Altitude band reward
        altitude = self.drone.position[2]
        optimal_min, optimal_max = 3.0, 8.0
        if optimal_min <= altitude <= optimal_max:
            reward += self.reward_weights["altitude_band"]
        else:
            deviation = min(abs(altitude - optimal_min), abs(altitude - optimal_max))
            reward += self.reward_weights["altitude_penalty"] * deviation

        # Lookahead alignment reward
        next_wp = self._get_waypoint(self.current_waypoint_idx + 1)
        if next_wp is not None:
            to_next = next_wp - self.drone.position
            to_next_norm = to_next / (np.linalg.norm(to_next) + 1e-8)

            # Check if velocity is aligned with next waypoint
            if np.linalg.norm(self.drone.velocity) > 0.1:
                vel_norm = self.drone.velocity / np.linalg.norm(self.drone.velocity)
                alignment = np.dot(vel_norm, to_next_norm)
                reward += self.reward_weights["lookahead_alignment"] * max(0, alignment)

        # Obstacle proximity penalty
        nearest_dist, _ = self.drone.get_nearest_obstacle_distance(self.obstacles)
        if nearest_dist < 3.0:
            proximity_penalty = (3.0 - nearest_dist) / 3.0
            reward += self.reward_weights["obstacle_proximity"] * proximity_penalty

        return reward

    def _in_bounds(self) -> bool:
        """Check if drone is within world bounds."""
        pos = self.drone.position
        return (self.config.x_bounds[0] <= pos[0] <= self.config.x_bounds[1] and
                self.config.y_bounds[0] <= pos[1] <= self.config.y_bounds[1] and
                self.config.z_bounds[0] <= pos[2] <= self.config.z_bounds[1])

    def _check_termination(self) -> Tuple[bool, bool]:
        """Check episode termination conditions.

        Returns:
            (terminated, truncated) tuple
        """
        # Terminated: goal reached or failure
        if self.drone.crashed:
            return True, False

        if self.drone.check_collision(self.obstacles, self.config.drone_radius):
            return True, False

        if not self._in_bounds():
            return True, False

        # Goal reached
        goal_distance = np.linalg.norm(self.drone.position - self.goal_position)
        if goal_distance < self.config.waypoint_reach_distance:
            if self.current_waypoint_idx >= len(self.route_waypoints) - 1:
                return True, False

        # Truncated: time limit
        if self.step_count >= self.config.max_episode_steps:
            return False, True

        return False, False

    def _get_info(self) -> Dict[str, Any]:
        """Get episode information."""
        return {
            "task": self.task.value,
            "difficulty": self.difficulty,
            "step_count": self.step_count,
            "current_waypoint_idx": self.current_waypoint_idx,
            "total_waypoints": len(self.route_waypoints),
            "distance_to_goal": float(np.linalg.norm(self.drone.position - self.goal_position)),
            "total_distance_traveled": self.total_distance_traveled,
            "drone_position": self.drone.position.tolist(),
            "drone_velocity": self.drone.velocity.tolist(),
            "crashed": self.drone.crashed,
        }

    def render(self):
        """Render the environment."""
        if self.render_mode is None:
            return None

        if self._renderer is None:
            from .visualization import DroneRenderer
            self._renderer = DroneRenderer(self)

        return self._renderer.render(mode=self.render_mode)

    def close(self):
        """Clean up resources."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


def make_env(
    task: str = "hover",
    difficulty: float = 0.5,
    render_mode: str = None,
    seed: int = None
) -> DroneEnv:
    """Factory function to create drone environment.

    Args:
        task: Task name ("hover", "delivery", "delivery_route")
        difficulty: Difficulty level 0.0-1.0
        render_mode: Rendering mode
        seed: Random seed

    Returns:
        Configured DroneEnv instance
    """
    task_type = TaskType(task)
    return DroneEnv(
        task=task_type,
        difficulty=difficulty,
        render_mode=render_mode,
        seed=seed
    )
