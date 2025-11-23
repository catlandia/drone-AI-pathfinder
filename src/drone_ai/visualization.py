"""Visualization for drone navigation environment.

Provides 2D and 3D rendering of drone, obstacles, waypoints, and paths.
"""

import numpy as np
from typing import List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .environment import DroneEnv

# Try to import visualization libraries
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False


class DroneRenderer:
    """Renderer for drone environment visualization."""

    def __init__(self, env: 'DroneEnv', width: int = 800, height: int = 600):
        """Initialize renderer.

        Args:
            env: DroneEnv instance to render
            width: Window width in pixels
            height: Window height in pixels
        """
        self.env = env
        self.width = width
        self.height = height

        self._fig = None
        self._ax = None
        self._pygame_screen = None
        self._initialized = False

    def _init_matplotlib(self):
        """Initialize matplotlib 3D figure."""
        if not MATPLOTLIB_AVAILABLE:
            return False

        plt.ion()
        self._fig = plt.figure(figsize=(10, 8))
        self._ax = self._fig.add_subplot(111, projection='3d')
        self._initialized = True
        return True

    def _init_pygame(self):
        """Initialize pygame display."""
        if not PYGAME_AVAILABLE:
            return False

        pygame.init()
        self._pygame_screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Drone Navigation")
        self._font = pygame.font.Font(None, 24)
        self._initialized = True
        return True

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """Render current environment state.

        Args:
            mode: "human" for display, "rgb_array" for pixel array

        Returns:
            RGB array if mode is "rgb_array", else None
        """
        if mode == "human":
            if MATPLOTLIB_AVAILABLE:
                return self._render_matplotlib()
            elif PYGAME_AVAILABLE:
                return self._render_pygame()
            else:
                return self._render_ascii()
        elif mode == "rgb_array":
            return self._render_to_array()

        return None

    def _render_matplotlib(self) -> None:
        """Render using matplotlib 3D."""
        if not self._initialized:
            if not self._init_matplotlib():
                return

        self._ax.clear()

        # Set axis limits
        config = self.env.config
        self._ax.set_xlim(config.x_bounds)
        self._ax.set_ylim(config.y_bounds)
        self._ax.set_zlim(config.z_bounds)

        self._ax.set_xlabel('X (m)')
        self._ax.set_ylabel('Y (m)')
        self._ax.set_zlabel('Z (m)')

        # Draw ground plane
        self._draw_ground_plane()

        # Draw obstacles
        self._draw_obstacles_3d()

        # Draw waypoints and path
        self._draw_path_3d()

        # Draw drone
        self._draw_drone_3d()

        # Draw info text
        self._draw_info_text()

        self._fig.canvas.draw()
        self._fig.canvas.flush_events()
        plt.pause(0.001)

    def _draw_ground_plane(self):
        """Draw ground plane."""
        config = self.env.config
        xx, yy = np.meshgrid(
            np.linspace(config.x_bounds[0], config.x_bounds[1], 5),
            np.linspace(config.y_bounds[0], config.y_bounds[1], 5)
        )
        zz = np.zeros_like(xx)
        self._ax.plot_surface(xx, yy, zz, alpha=0.1, color='green')

    def _draw_obstacles_3d(self):
        """Draw obstacles in 3D."""
        from .simulation import ObstacleType

        for obstacle in self.env.obstacles:
            pos = obstacle.position

            if obstacle.obstacle_type == ObstacleType.SPHERE:
                radius = obstacle.size[0]
                u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
                x = pos[0] + radius * np.cos(u) * np.sin(v)
                y = pos[1] + radius * np.sin(u) * np.sin(v)
                z = pos[2] + radius * np.cos(v)
                self._ax.plot_surface(x, y, z, color='red', alpha=0.6)

            elif obstacle.obstacle_type == ObstacleType.CYLINDER:
                radius = obstacle.size[0]
                height = obstacle.size[1]
                z_cyl = np.linspace(pos[2], pos[2] + height, 10)
                theta = np.linspace(0, 2*np.pi, 20)
                theta_grid, z_grid = np.meshgrid(theta, z_cyl)
                x = pos[0] + radius * np.cos(theta_grid)
                y = pos[1] + radius * np.sin(theta_grid)
                self._ax.plot_surface(x, y, z_grid, color='orange', alpha=0.6)

            else:  # BOX
                self._draw_box_3d(pos, obstacle.size)

    def _draw_box_3d(self, center: np.ndarray, size: np.ndarray):
        """Draw a 3D box."""
        half = size / 2

        vertices = np.array([
            center + [-half[0], -half[1], -half[2]],
            center + [half[0], -half[1], -half[2]],
            center + [half[0], half[1], -half[2]],
            center + [-half[0], half[1], -half[2]],
            center + [-half[0], -half[1], half[2]],
            center + [half[0], -half[1], half[2]],
            center + [half[0], half[1], half[2]],
            center + [-half[0], half[1], half[2]],
        ])

        faces = [
            [vertices[j] for j in [0, 1, 2, 3]],
            [vertices[j] for j in [4, 5, 6, 7]],
            [vertices[j] for j in [0, 1, 5, 4]],
            [vertices[j] for j in [2, 3, 7, 6]],
            [vertices[j] for j in [0, 3, 7, 4]],
            [vertices[j] for j in [1, 2, 6, 5]],
        ]

        self._ax.add_collection3d(Poly3DCollection(
            faces, alpha=0.5, facecolor='purple', edgecolor='black'
        ))

    def _draw_path_3d(self):
        """Draw waypoints and connecting path."""
        waypoints = self.env.route_waypoints
        if not waypoints:
            return

        # Draw all waypoints
        for i, wp in enumerate(waypoints):
            if i < self.env.current_waypoint_idx:
                # Reached waypoints - gray
                color = 'gray'
                size = 30
            elif i == self.env.current_waypoint_idx:
                # Current target - green
                color = 'lime'
                size = 100
            else:
                # Future waypoints - cyan
                color = 'cyan'
                size = 50

            self._ax.scatter(*wp, c=color, s=size, marker='o', edgecolors='black')
            self._ax.text(wp[0], wp[1], wp[2] + 0.5, f'WP{i}', fontsize=8)

        # Draw path lines
        all_points = [self.env.drone.position] + waypoints
        for i in range(len(all_points) - 1):
            p1, p2 = all_points[i], all_points[i + 1]
            alpha = 0.3 if i < self.env.current_waypoint_idx else 0.8
            self._ax.plot(
                [p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                'b--', alpha=alpha, linewidth=2
            )

        # Draw goal marker
        goal = self.env.goal_position
        self._ax.scatter(*goal, c='gold', s=200, marker='*', edgecolors='black')

    def _draw_drone_3d(self):
        """Draw drone at current position."""
        pos = self.env.drone.position
        vel = self.env.drone.velocity

        # Drone body
        self._ax.scatter(*pos, c='blue', s=150, marker='^')

        # Velocity vector
        if np.linalg.norm(vel) > 0.1:
            vel_scaled = vel / np.linalg.norm(vel) * 2
            self._ax.quiver(
                pos[0], pos[1], pos[2],
                vel_scaled[0], vel_scaled[1], vel_scaled[2],
                color='red', arrow_length_ratio=0.3
            )

        # Heading indicator
        yaw = self.env.drone.orientation[2]
        heading = np.array([np.cos(yaw), np.sin(yaw), 0]) * 1.5
        self._ax.quiver(
            pos[0], pos[1], pos[2],
            heading[0], heading[1], heading[2],
            color='green', arrow_length_ratio=0.3
        )

    def _draw_info_text(self):
        """Draw information text on plot."""
        info = self.env._get_info()
        text = (
            f"Task: {info['task']}\n"
            f"Difficulty: {info['difficulty']:.2f}\n"
            f"Step: {info['step_count']}\n"
            f"Waypoint: {info['current_waypoint_idx'] + 1}/{info['total_waypoints']}\n"
            f"Distance to Goal: {info['distance_to_goal']:.1f}m"
        )
        self._ax.text2D(0.02, 0.98, text, transform=self._ax.transAxes,
                        fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    def _render_pygame(self) -> None:
        """Render using pygame (2D top-down view)."""
        if not self._initialized:
            if not self._init_pygame():
                return

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

        # Clear screen
        self._pygame_screen.fill((200, 230, 200))  # Light green background

        config = self.env.config

        # Coordinate transform functions
        def world_to_screen(x, y):
            screen_x = int((x - config.x_bounds[0]) / (config.x_bounds[1] - config.x_bounds[0]) * self.width)
            screen_y = int((1 - (y - config.y_bounds[0]) / (config.y_bounds[1] - config.y_bounds[0])) * self.height)
            return screen_x, screen_y

        def size_to_screen(size):
            return int(size / (config.x_bounds[1] - config.x_bounds[0]) * self.width)

        # Draw obstacles
        for obstacle in self.env.obstacles:
            pos = obstacle.position
            screen_pos = world_to_screen(pos[0], pos[1])
            radius = size_to_screen(obstacle.get_radius())
            pygame.draw.circle(self._pygame_screen, (200, 50, 50), screen_pos, radius)
            pygame.draw.circle(self._pygame_screen, (100, 25, 25), screen_pos, radius, 2)

        # Draw path
        waypoints = self.env.route_waypoints
        if waypoints:
            points = [world_to_screen(wp[0], wp[1]) for wp in waypoints]

            # Draw lines
            if len(points) > 1:
                pygame.draw.lines(self._pygame_screen, (50, 50, 200), False, points, 2)

            # Draw waypoints
            for i, point in enumerate(points):
                if i < self.env.current_waypoint_idx:
                    color = (128, 128, 128)
                elif i == self.env.current_waypoint_idx:
                    color = (0, 255, 0)
                else:
                    color = (0, 200, 200)
                pygame.draw.circle(self._pygame_screen, color, point, 8)

        # Draw goal
        goal_screen = world_to_screen(self.env.goal_position[0], self.env.goal_position[1])
        pygame.draw.polygon(self._pygame_screen, (255, 215, 0), [
            (goal_screen[0], goal_screen[1] - 15),
            (goal_screen[0] - 10, goal_screen[1] + 10),
            (goal_screen[0] + 10, goal_screen[1] + 10),
        ])

        # Draw drone
        drone_pos = self.env.drone.position
        drone_screen = world_to_screen(drone_pos[0], drone_pos[1])
        pygame.draw.circle(self._pygame_screen, (0, 0, 255), drone_screen, 12)

        # Draw heading
        yaw = self.env.drone.orientation[2]
        heading_end = (
            drone_screen[0] + int(20 * np.cos(-yaw + np.pi/2)),
            drone_screen[1] + int(20 * np.sin(-yaw + np.pi/2))
        )
        pygame.draw.line(self._pygame_screen, (0, 255, 0), drone_screen, heading_end, 3)

        # Draw info text
        info = self.env._get_info()
        lines = [
            f"Task: {info['task']}",
            f"Difficulty: {info['difficulty']:.2f}",
            f"Step: {info['step_count']}",
            f"Waypoint: {info['current_waypoint_idx'] + 1}/{info['total_waypoints']}",
            f"Altitude: {drone_pos[2]:.1f}m",
            f"To Goal: {info['distance_to_goal']:.1f}m",
        ]
        for i, line in enumerate(lines):
            text = self._font.render(line, True, (0, 0, 0))
            self._pygame_screen.blit(text, (10, 10 + i * 20))

        pygame.display.flip()

    def _render_ascii(self) -> None:
        """Simple ASCII rendering fallback."""
        config = self.env.config
        width, height = 60, 30

        # Create grid
        grid = [['.' for _ in range(width)] for _ in range(height)]

        def world_to_grid(x, y):
            gx = int((x - config.x_bounds[0]) / (config.x_bounds[1] - config.x_bounds[0]) * (width - 1))
            gy = int((1 - (y - config.y_bounds[0]) / (config.y_bounds[1] - config.y_bounds[0])) * (height - 1))
            return max(0, min(width - 1, gx)), max(0, min(height - 1, gy))

        # Draw obstacles
        for obstacle in self.env.obstacles:
            gx, gy = world_to_grid(obstacle.position[0], obstacle.position[1])
            grid[gy][gx] = 'O'

        # Draw waypoints
        for i, wp in enumerate(self.env.route_waypoints):
            gx, gy = world_to_grid(wp[0], wp[1])
            if i == self.env.current_waypoint_idx:
                grid[gy][gx] = '@'
            else:
                grid[gy][gx] = str(i % 10)

        # Draw drone
        gx, gy = world_to_grid(self.env.drone.position[0], self.env.drone.position[1])
        grid[gy][gx] = 'D'

        # Print
        print('\n' + '=' * width)
        for row in grid:
            print(''.join(row))
        print('=' * width)

        info = self.env._get_info()
        print(f"Step: {info['step_count']} | "
              f"WP: {info['current_waypoint_idx'] + 1}/{info['total_waypoints']} | "
              f"Alt: {self.env.drone.position[2]:.1f}m | "
              f"Dist: {info['distance_to_goal']:.1f}m")

    def _render_to_array(self) -> np.ndarray:
        """Render to RGB array."""
        if not MATPLOTLIB_AVAILABLE:
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)

        if not self._initialized:
            self._init_matplotlib()

        self._render_matplotlib()

        # Get array from figure
        self._fig.canvas.draw()
        buf = self._fig.canvas.tostring_rgb()
        ncols, nrows = self._fig.canvas.get_width_height()
        return np.frombuffer(buf, dtype=np.uint8).reshape(nrows, ncols, 3)

    def close(self):
        """Close rendering resources."""
        if self._fig is not None:
            plt.close(self._fig)
            self._fig = None
            self._ax = None

        if self._pygame_screen is not None:
            pygame.quit()
            self._pygame_screen = None

        self._initialized = False


def visualize_path(
    waypoints: List[np.ndarray],
    obstacles: List = None,
    start: np.ndarray = None,
    goal: np.ndarray = None,
    title: str = "Path Visualization"
):
    """Standalone function to visualize a path.

    Args:
        waypoints: List of waypoint positions
        obstacles: Optional list of obstacles
        start: Optional start position
        goal: Optional goal position
        title: Plot title
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available for visualization")
        return

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Draw waypoints
    if waypoints:
        points = np.array(waypoints)
        ax.plot(points[:, 0], points[:, 1], points[:, 2], 'b-o', label='Path')

    # Draw start
    if start is not None:
        ax.scatter(*start, c='green', s=200, marker='o', label='Start')

    # Draw goal
    if goal is not None:
        ax.scatter(*goal, c='red', s=200, marker='*', label='Goal')

    # Draw obstacles
    if obstacles:
        for obs in obstacles:
            ax.scatter(*obs.position, c='red', s=100, marker='s', alpha=0.5)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title)
    ax.legend()

    plt.show()
