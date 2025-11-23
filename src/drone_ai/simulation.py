"""Drone physics simulation and obstacle definitions.

This module provides the core simulation components:
- Drone: Quadrotor dynamics with realistic physics
- Obstacle: Various obstacle types for collision detection
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from enum import Enum


class ObstacleType(Enum):
    """Types of obstacles in the environment."""
    SPHERE = "sphere"
    CYLINDER = "cylinder"
    BOX = "box"


@dataclass
class Obstacle:
    """Represents an obstacle in the environment.

    Attributes:
        position: Center position [x, y, z] in meters
        size: Dimensions depending on type:
            - SPHERE: [radius]
            - CYLINDER: [radius, height]
            - BOX: [width, depth, height]
        obstacle_type: Type of obstacle geometry
        velocity: Optional velocity for moving obstacles [vx, vy, vz]
    """
    position: np.ndarray
    size: np.ndarray
    obstacle_type: ObstacleType = ObstacleType.SPHERE
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self):
        self.position = np.array(self.position, dtype=np.float32)
        self.size = np.array(self.size, dtype=np.float32)
        self.velocity = np.array(self.velocity, dtype=np.float32)

    def get_radius(self) -> float:
        """Get the effective bounding radius of the obstacle."""
        if self.obstacle_type == ObstacleType.SPHERE:
            return float(self.size[0])
        elif self.obstacle_type == ObstacleType.CYLINDER:
            return float(self.size[0])
        else:  # BOX
            return float(np.linalg.norm(self.size[:2]) / 2)

    def get_height(self) -> float:
        """Get the height of the obstacle."""
        if self.obstacle_type == ObstacleType.SPHERE:
            return float(self.size[0] * 2)
        elif self.obstacle_type == ObstacleType.CYLINDER:
            return float(self.size[1])
        else:  # BOX
            return float(self.size[2])

    def contains_point(self, point: np.ndarray, margin: float = 0.0) -> bool:
        """Check if a point is inside the obstacle (with optional safety margin)."""
        point = np.array(point)

        if self.obstacle_type == ObstacleType.SPHERE:
            dist = np.linalg.norm(point - self.position)
            return dist < (self.size[0] + margin)

        elif self.obstacle_type == ObstacleType.CYLINDER:
            # Check horizontal distance
            horiz_dist = np.linalg.norm(point[:2] - self.position[:2])
            # Check vertical bounds
            bottom = self.position[2]
            top = self.position[2] + self.size[1]
            in_vertical = bottom - margin <= point[2] <= top + margin
            return horiz_dist < (self.size[0] + margin) and in_vertical

        else:  # BOX
            half_size = self.size / 2 + margin
            rel_pos = np.abs(point - self.position)
            return np.all(rel_pos < half_size)

    def distance_to_point(self, point: np.ndarray) -> float:
        """Calculate minimum distance from point to obstacle surface."""
        point = np.array(point)

        if self.obstacle_type == ObstacleType.SPHERE:
            dist = np.linalg.norm(point - self.position) - self.size[0]
            return max(0.0, dist)

        elif self.obstacle_type == ObstacleType.CYLINDER:
            # Horizontal distance to cylinder axis
            horiz_dist = np.linalg.norm(point[:2] - self.position[:2])
            horiz_penetration = horiz_dist - self.size[0]

            # Vertical distance
            bottom = self.position[2]
            top = self.position[2] + self.size[1]
            if point[2] < bottom:
                vert_dist = bottom - point[2]
            elif point[2] > top:
                vert_dist = point[2] - top
            else:
                vert_dist = 0.0

            if horiz_penetration > 0 and vert_dist > 0:
                return np.sqrt(horiz_penetration**2 + vert_dist**2)
            else:
                return max(horiz_penetration, vert_dist, 0.0)

        else:  # BOX
            half_size = self.size / 2
            rel_pos = point - self.position
            closest = np.clip(rel_pos, -half_size, half_size)
            return float(np.linalg.norm(rel_pos - closest))

    def update(self, dt: float):
        """Update obstacle position for moving obstacles."""
        self.position = self.position + self.velocity * dt


@dataclass
class DroneState:
    """Complete state of the drone."""
    position: np.ndarray  # [x, y, z] in meters
    velocity: np.ndarray  # [vx, vy, vz] in m/s
    orientation: np.ndarray  # [roll, pitch, yaw] in radians
    angular_velocity: np.ndarray  # [p, q, r] in rad/s


class Drone:
    """Simulated quadrotor drone with realistic physics.

    Uses a simplified quadrotor model with:
    - Mass and inertia properties
    - Thrust and torque from 4 rotors
    - Aerodynamic drag
    - Gravity
    """

    # Physical properties
    MASS = 1.0  # kg
    ARM_LENGTH = 0.25  # meters
    INERTIA = np.diag([0.01, 0.01, 0.02])  # kg*m^2

    # Motor properties
    MAX_THRUST = 15.0  # N (total)
    MAX_TORQUE = 1.0  # N*m

    # Aerodynamic properties
    DRAG_COEFF = 0.1  # Linear drag coefficient

    # Environment
    GRAVITY = 9.81  # m/s^2

    def __init__(self, position: np.ndarray = None):
        """Initialize drone at given position."""
        self.position = np.array(position if position is not None else [0, 0, 1], dtype=np.float32)
        self.velocity = np.zeros(3, dtype=np.float32)
        self.orientation = np.zeros(3, dtype=np.float32)  # roll, pitch, yaw
        self.angular_velocity = np.zeros(3, dtype=np.float32)

        # Motor speeds (normalized 0-1)
        self.motor_speeds = np.ones(4) * 0.5

        # Collision state
        self.crashed = False

    def get_state(self) -> DroneState:
        """Get current drone state."""
        return DroneState(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            orientation=self.orientation.copy(),
            angular_velocity=self.angular_velocity.copy()
        )

    def set_state(self, state: DroneState):
        """Set drone state."""
        self.position = state.position.copy()
        self.velocity = state.velocity.copy()
        self.orientation = state.orientation.copy()
        self.angular_velocity = state.angular_velocity.copy()

    def reset(self, position: np.ndarray = None, orientation: np.ndarray = None):
        """Reset drone to initial state."""
        self.position = np.array(position if position is not None else [0, 0, 1], dtype=np.float32)
        self.velocity = np.zeros(3, dtype=np.float32)
        self.orientation = np.array(orientation if orientation is not None else [0, 0, 0], dtype=np.float32)
        self.angular_velocity = np.zeros(3, dtype=np.float32)
        self.motor_speeds = np.ones(4) * 0.5
        self.crashed = False

    def _rotation_matrix(self) -> np.ndarray:
        """Compute rotation matrix from orientation (roll, pitch, yaw)."""
        roll, pitch, yaw = self.orientation

        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)

        R = np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp, cp*sr, cp*cr]
        ])
        return R

    def apply_action(self, action: np.ndarray, dt: float):
        """Apply control action and simulate one timestep.

        Args:
            action: [thrust, roll_rate, pitch_rate, yaw_rate] normalized to [-1, 1]
            dt: Timestep in seconds
        """
        if self.crashed:
            return

        action = np.clip(action, -1, 1)

        # Extract action components
        thrust_cmd = (action[0] + 1) / 2  # Map [-1,1] to [0,1]
        roll_rate_cmd = action[1] * self.MAX_TORQUE
        pitch_rate_cmd = action[2] * self.MAX_TORQUE
        yaw_rate_cmd = action[3] * self.MAX_TORQUE

        # Compute thrust in body frame (upward)
        thrust_magnitude = thrust_cmd * self.MAX_THRUST
        thrust_body = np.array([0, 0, thrust_magnitude])

        # Rotate thrust to world frame
        R = self._rotation_matrix()
        thrust_world = R @ thrust_body

        # Compute acceleration
        gravity = np.array([0, 0, -self.GRAVITY * self.MASS])
        drag = -self.DRAG_COEFF * self.velocity * np.abs(self.velocity)

        acceleration = (thrust_world + gravity + drag) / self.MASS

        # Update velocity and position
        self.velocity = self.velocity + acceleration * dt
        self.position = self.position + self.velocity * dt

        # Update angular velocity with simple rate control
        target_angular_vel = np.array([roll_rate_cmd, pitch_rate_cmd, yaw_rate_cmd])
        self.angular_velocity = self.angular_velocity + (target_angular_vel - self.angular_velocity) * 5.0 * dt

        # Update orientation
        self.orientation = self.orientation + self.angular_velocity * dt

        # Wrap yaw to [-pi, pi]
        self.orientation[2] = np.arctan2(np.sin(self.orientation[2]), np.cos(self.orientation[2]))

        # Clamp roll and pitch
        self.orientation[0] = np.clip(self.orientation[0], -np.pi/4, np.pi/4)
        self.orientation[1] = np.clip(self.orientation[1], -np.pi/4, np.pi/4)

        # Check ground collision
        if self.position[2] < 0:
            self.position[2] = 0
            self.velocity[2] = 0
            if np.linalg.norm(self.velocity) > 2.0:
                self.crashed = True

    def check_collision(self, obstacles: List[Obstacle], margin: float = 0.3) -> bool:
        """Check if drone collides with any obstacle.

        Args:
            obstacles: List of obstacles to check
            margin: Safety margin around drone (drone radius)

        Returns:
            True if collision detected
        """
        for obstacle in obstacles:
            if obstacle.contains_point(self.position, margin):
                self.crashed = True
                return True
        return False

    def get_nearest_obstacle_distance(self, obstacles: List[Obstacle]) -> Tuple[float, Optional[Obstacle]]:
        """Find distance to nearest obstacle.

        Returns:
            Tuple of (distance, obstacle) or (inf, None) if no obstacles
        """
        min_dist = float('inf')
        nearest = None

        for obstacle in obstacles:
            dist = obstacle.distance_to_point(self.position)
            if dist < min_dist:
                min_dist = dist
                nearest = obstacle

        return min_dist, nearest

    def get_obstacle_distances_in_direction(
        self,
        obstacles: List[Obstacle],
        direction: np.ndarray,
        cone_angle: float = np.pi/4
    ) -> List[Tuple[float, Obstacle]]:
        """Get distances to obstacles in a given direction (within cone).

        Args:
            obstacles: List of obstacles
            direction: Direction vector to check
            cone_angle: Half-angle of detection cone in radians

        Returns:
            List of (distance, obstacle) tuples sorted by distance
        """
        direction = direction / (np.linalg.norm(direction) + 1e-8)
        results = []

        for obstacle in obstacles:
            to_obstacle = obstacle.position - self.position
            dist = np.linalg.norm(to_obstacle)
            if dist < 1e-6:
                continue

            to_obstacle_norm = to_obstacle / dist
            angle = np.arccos(np.clip(np.dot(direction, to_obstacle_norm), -1, 1))

            if angle < cone_angle:
                actual_dist = obstacle.distance_to_point(self.position)
                results.append((actual_dist, obstacle))

        return sorted(results, key=lambda x: x[0])


def generate_random_obstacles(
    num_obstacles: int,
    bounds: Tuple[float, float, float, float, float, float],
    min_size: float = 0.5,
    max_size: float = 2.0,
    seed: int = None
) -> List[Obstacle]:
    """Generate random obstacles within bounds.

    Args:
        num_obstacles: Number of obstacles to generate
        bounds: (x_min, x_max, y_min, y_max, z_min, z_max)
        min_size: Minimum obstacle size
        max_size: Maximum obstacle size
        seed: Random seed for reproducibility

    Returns:
        List of Obstacle objects
    """
    if seed is not None:
        np.random.seed(seed)

    obstacles = []
    x_min, x_max, y_min, y_max, z_min, z_max = bounds

    for _ in range(num_obstacles):
        # Random position within bounds
        position = np.array([
            np.random.uniform(x_min, x_max),
            np.random.uniform(y_min, y_max),
            np.random.uniform(z_min, z_max)
        ])

        # Random type
        obstacle_type = np.random.choice(list(ObstacleType))

        # Random size based on type
        if obstacle_type == ObstacleType.SPHERE:
            size = np.array([np.random.uniform(min_size, max_size)])
        elif obstacle_type == ObstacleType.CYLINDER:
            radius = np.random.uniform(min_size, max_size)
            height = np.random.uniform(min_size * 2, max_size * 3)
            size = np.array([radius, height])
        else:  # BOX
            size = np.random.uniform(min_size, max_size, size=3)

        obstacles.append(Obstacle(position, size, obstacle_type))

    return obstacles
