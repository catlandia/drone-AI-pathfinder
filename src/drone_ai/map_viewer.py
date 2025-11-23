"""3D Movement Map Viewer for Drone AI.

Visualizes drone movement in real-time with:
- 3D trajectory trail
- Waypoints and obstacles
- Altitude color coding
- Live position updates

Run with:
    python -m drone_ai.map_viewer
    python -m drone_ai.map_viewer --task delivery_route --difficulty 0.7
"""

import numpy as np
import argparse
import time
from typing import List, Optional

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.animation import FuncAnimation
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("matplotlib required: pip install matplotlib")

from .environment import DroneEnv, TaskType
from .simulation import Obstacle, ObstacleType


class DroneMapViewer:
    """Interactive 3D map viewer showing drone movement."""

    def __init__(
        self,
        env: DroneEnv,
        trail_length: int = 200,
        update_interval: int = 50
    ):
        """Initialize map viewer.

        Args:
            env: DroneEnv instance
            trail_length: Number of past positions to show in trail
            update_interval: Milliseconds between updates
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib is required for DroneMapViewer. Install with: pip install matplotlib")

        self.env = env
        self.trail_length = trail_length
        self.update_interval = update_interval

        # Position history for trail
        self.position_history: List[np.ndarray] = []
        self.altitude_history: List[float] = []

        # Setup figure
        self.fig = plt.figure(figsize=(14, 10))

        # Main 3D view
        self.ax_3d = self.fig.add_subplot(2, 2, 1, projection='3d')

        # Top-down view (XY)
        self.ax_top = self.fig.add_subplot(2, 2, 2)

        # Side view (XZ)
        self.ax_side = self.fig.add_subplot(2, 2, 3)

        # Info panel
        self.ax_info = self.fig.add_subplot(2, 2, 4)
        self.ax_info.axis('off')

        self.fig.suptitle('Drone AI - 3D Movement Map', fontsize=14, fontweight='bold')

        # Animation
        self.anim = None
        self.running = True
        self.step_count = 0

    def _draw_obstacles(self, ax, projection='3d'):
        """Draw obstacles on axis."""
        for obs in self.env.obstacles:
            pos = obs.position
            radius = obs.get_radius()

            if projection == '3d':
                # Draw sphere approximation
                u, v = np.mgrid[0:2*np.pi:12j, 0:np.pi:8j]
                x = pos[0] + radius * np.cos(u) * np.sin(v)
                y = pos[1] + radius * np.sin(u) * np.sin(v)
                z = pos[2] + radius * np.cos(v)
                ax.plot_surface(x, y, z, color='red', alpha=0.4)

            elif projection == 'xy':
                circle = plt.Circle((pos[0], pos[1]), radius, color='red', alpha=0.5)
                ax.add_patch(circle)

            elif projection == 'xz':
                # Side view - show as rectangle
                height = obs.get_height()
                rect = plt.Rectangle(
                    (pos[0] - radius, pos[2]),
                    radius * 2, height,
                    color='red', alpha=0.5
                )
                ax.add_patch(rect)

    def _draw_waypoints(self, ax, projection='3d'):
        """Draw waypoints on axis."""
        waypoints = self.env.route_waypoints

        for i, wp in enumerate(waypoints):
            if i < self.env.current_waypoint_idx:
                color = 'gray'
                size = 30
            elif i == self.env.current_waypoint_idx:
                color = 'lime'
                size = 100
            else:
                color = 'cyan'
                size = 50

            if projection == '3d':
                ax.scatter(wp[0], wp[1], wp[2], c=color, s=size, marker='o', edgecolors='black')
            elif projection == 'xy':
                ax.scatter(wp[0], wp[1], c=color, s=size, marker='o', edgecolors='black')
            elif projection == 'xz':
                ax.scatter(wp[0], wp[2], c=color, s=size, marker='o', edgecolors='black')

        # Draw path lines
        if len(waypoints) > 1:
            wps = np.array(waypoints)
            if projection == '3d':
                ax.plot(wps[:, 0], wps[:, 1], wps[:, 2], 'b--', alpha=0.5, linewidth=1)
            elif projection == 'xy':
                ax.plot(wps[:, 0], wps[:, 1], 'b--', alpha=0.5, linewidth=1)
            elif projection == 'xz':
                ax.plot(wps[:, 0], wps[:, 2], 'b--', alpha=0.5, linewidth=1)

    def _draw_trail(self, ax, projection='3d'):
        """Draw drone movement trail."""
        if len(self.position_history) < 2:
            return

        positions = np.array(self.position_history)
        altitudes = np.array(self.altitude_history)

        # Color based on altitude (blue=low, red=high)
        alt_norm = (altitudes - altitudes.min()) / (altitudes.max() - altitudes.min() + 0.01)

        if projection == '3d':
            # Draw colored trail segments
            for i in range(len(positions) - 1):
                color = plt.cm.coolwarm(alt_norm[i])
                ax.plot(
                    positions[i:i+2, 0],
                    positions[i:i+2, 1],
                    positions[i:i+2, 2],
                    color=color, linewidth=2, alpha=0.7
                )
        elif projection == 'xy':
            for i in range(len(positions) - 1):
                color = plt.cm.coolwarm(alt_norm[i])
                ax.plot(
                    positions[i:i+2, 0],
                    positions[i:i+2, 1],
                    color=color, linewidth=2, alpha=0.7
                )
        elif projection == 'xz':
            for i in range(len(positions) - 1):
                color = plt.cm.coolwarm(alt_norm[i])
                ax.plot(
                    positions[i:i+2, 0],
                    positions[i:i+2, 2],
                    color=color, linewidth=2, alpha=0.7
                )

    def _draw_drone(self, ax, projection='3d'):
        """Draw drone at current position."""
        pos = self.env.drone.position
        vel = self.env.drone.velocity

        if projection == '3d':
            ax.scatter(pos[0], pos[1], pos[2], c='blue', s=200, marker='^', edgecolors='white', zorder=10)
            # Velocity vector
            if np.linalg.norm(vel) > 0.1:
                vel_scaled = vel / (np.linalg.norm(vel) + 0.01) * 2
                ax.quiver(pos[0], pos[1], pos[2], vel_scaled[0], vel_scaled[1], vel_scaled[2],
                         color='yellow', arrow_length_ratio=0.3, linewidth=2)
        elif projection == 'xy':
            ax.scatter(pos[0], pos[1], c='blue', s=200, marker='^', edgecolors='white', zorder=10)
            # Heading arrow
            yaw = self.env.drone.orientation[2]
            ax.arrow(pos[0], pos[1], np.cos(yaw)*2, np.sin(yaw)*2,
                    head_width=0.5, head_length=0.3, fc='green', ec='green')
        elif projection == 'xz':
            ax.scatter(pos[0], pos[2], c='blue', s=200, marker='^', edgecolors='white', zorder=10)

    def _update_info(self):
        """Update info panel."""
        self.ax_info.clear()
        self.ax_info.axis('off')

        info = self.env._get_info()
        pos = self.env.drone.position
        vel = self.env.drone.velocity

        text = f"""
DRONE STATUS
═══════════════════════════

Position:
  X: {pos[0]:7.2f} m
  Y: {pos[1]:7.2f} m
  Z: {pos[2]:7.2f} m (altitude)

Velocity:
  Speed: {np.linalg.norm(vel):5.2f} m/s

MISSION
═══════════════════════════

Task: {info['task']}
Difficulty: {info['difficulty']:.1f}

Waypoint: {info['current_waypoint_idx'] + 1} / {info['total_waypoints']}
Distance to goal: {info['distance_to_goal']:.1f} m

Step: {self.step_count}

LEGEND
═══════════════════════════
▲ Drone (blue)
● Current waypoint (green)
● Future waypoints (cyan)
● Completed waypoints (gray)
■ Obstacles (red)
━ Trail (blue=low, red=high)
        """

        self.ax_info.text(0.05, 0.95, text, transform=self.ax_info.transAxes,
                         fontsize=9, verticalalignment='top', fontfamily='monospace',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    def _update_frame(self, frame):
        """Update animation frame."""
        if not self.running:
            return

        # Take action (simple reactive controller)
        obs = self.env._get_observation()
        action = self._compute_action(obs)

        # Step environment
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.step_count += 1

        # Record position
        self.position_history.append(self.env.drone.position.copy())
        self.altitude_history.append(self.env.drone.position[2])

        # Limit trail length
        if len(self.position_history) > self.trail_length:
            self.position_history.pop(0)
            self.altitude_history.pop(0)

        # Reset if episode ends
        if terminated or truncated:
            self.env.reset()
            self.position_history.clear()
            self.altitude_history.clear()

        # Redraw all axes
        self._redraw()

    def _redraw(self):
        """Redraw all views."""
        config = self.env.config

        # Clear all axes
        self.ax_3d.clear()
        self.ax_top.clear()
        self.ax_side.clear()

        # 3D View
        self.ax_3d.set_xlim(config.x_bounds)
        self.ax_3d.set_ylim(config.y_bounds)
        self.ax_3d.set_zlim(config.z_bounds)
        self.ax_3d.set_xlabel('X (m)')
        self.ax_3d.set_ylabel('Y (m)')
        self.ax_3d.set_zlabel('Altitude (m)')
        self.ax_3d.set_title('3D View')

        self._draw_obstacles(self.ax_3d, '3d')
        self._draw_waypoints(self.ax_3d, '3d')
        self._draw_trail(self.ax_3d, '3d')
        self._draw_drone(self.ax_3d, '3d')

        # Top-down view
        self.ax_top.set_xlim(config.x_bounds)
        self.ax_top.set_ylim(config.y_bounds)
        self.ax_top.set_xlabel('X (m)')
        self.ax_top.set_ylabel('Y (m)')
        self.ax_top.set_title('Top View (XY)')
        self.ax_top.set_aspect('equal')
        self.ax_top.grid(True, alpha=0.3)

        self._draw_obstacles(self.ax_top, 'xy')
        self._draw_waypoints(self.ax_top, 'xy')
        self._draw_trail(self.ax_top, 'xy')
        self._draw_drone(self.ax_top, 'xy')

        # Side view
        self.ax_side.set_xlim(config.x_bounds)
        self.ax_side.set_ylim(config.z_bounds)
        self.ax_side.set_xlabel('X (m)')
        self.ax_side.set_ylabel('Altitude (m)')
        self.ax_side.set_title('Side View (XZ)')
        self.ax_side.grid(True, alpha=0.3)

        self._draw_obstacles(self.ax_side, 'xz')
        self._draw_waypoints(self.ax_side, 'xz')
        self._draw_trail(self.ax_side, 'xz')
        self._draw_drone(self.ax_side, 'xz')

        # Info panel
        self._update_info()

        self.fig.tight_layout()

    def _compute_action(self, obs: np.ndarray) -> np.ndarray:
        """Simple reactive controller."""
        position = obs[0:3]
        velocity = obs[3:6]
        orientation = obs[6:9]
        to_waypoint = obs[9:12]
        distance = obs[12]

        # Altitude control
        altitude_error = to_waypoint[2]
        thrust = 0.5 * altitude_error - 0.3 * velocity[2]
        thrust = np.clip(thrust, -0.5, 0.8)

        # Yaw control
        if np.linalg.norm(to_waypoint[:2]) > 0.1:
            desired_yaw = np.arctan2(to_waypoint[1], to_waypoint[0])
        else:
            desired_yaw = orientation[2]

        yaw_error = desired_yaw - orientation[2]
        yaw_error = np.arctan2(np.sin(yaw_error), np.cos(yaw_error))
        yaw_rate = np.clip(yaw_error * 2.0, -1, 1)

        # Forward control
        alignment = np.cos(yaw_error)
        if alignment > 0.5:
            forward_speed = min(2.0, distance * 0.3)
            current_forward = np.dot(velocity[:2], np.array([np.cos(orientation[2]), np.sin(orientation[2])]))
            pitch_rate = -0.3 * (forward_speed - current_forward)
        else:
            pitch_rate = 0.0
        pitch_rate = np.clip(pitch_rate, -0.5, 0.5)

        roll_rate = -2.0 * orientation[0]

        return np.array([thrust, roll_rate, pitch_rate, yaw_rate], dtype=np.float32)

    def run(self):
        """Run the interactive map viewer."""
        if not MATPLOTLIB_AVAILABLE:
            print("matplotlib is required for 3D map viewer")
            return

        print("\n" + "=" * 50)
        print("  DRONE AI - 3D MOVEMENT MAP")
        print("=" * 50)
        print("\nControls:")
        print("  - Close window to exit")
        print("  - Rotate 3D view with mouse drag")
        print("\nWatching drone navigate...")
        print("=" * 50 + "\n")

        # Initial draw
        self.env.reset()
        self._redraw()

        # Start animation
        self.anim = FuncAnimation(
            self.fig,
            self._update_frame,
            interval=self.update_interval,
            blit=False,
            cache_frame_data=False
        )

        plt.show()

    def close(self):
        """Close viewer."""
        self.running = False
        if self.anim:
            self.anim.event_source.stop()
        plt.close(self.fig)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Drone AI 3D Movement Map")
    parser.add_argument("--task", default="delivery_route",
                        choices=["hover", "delivery", "delivery_route"])
    parser.add_argument("--difficulty", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trail", type=int, default=200,
                        help="Trail length (number of past positions)")

    args = parser.parse_args()

    if not MATPLOTLIB_AVAILABLE:
        print("This feature requires matplotlib.")
        print("Install with: pip install matplotlib")
        return

    # Create environment
    env = DroneEnv(
        task=TaskType(args.task),
        difficulty=args.difficulty,
        seed=args.seed
    )

    # Create and run viewer
    viewer = DroneMapViewer(env, trail_length=args.trail)

    try:
        viewer.run()
    finally:
        viewer.close()
        env.close()


if __name__ == "__main__":
    main()
