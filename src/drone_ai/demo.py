"""Demo script for drone navigation environment.

Run with:
    python -m drone_ai.demo --task delivery_route --difficulty 0.5
"""

import argparse
import numpy as np
import time
from typing import Optional

from .environment import DroneEnv, TaskType, make_env


def run_demo(
    task: str = "delivery_route",
    difficulty: float = 0.5,
    render: bool = True,
    num_episodes: int = 3,
    max_steps: int = 500,
    seed: Optional[int] = None
):
    """Run demo of drone environment.

    Args:
        task: Task type ("hover", "delivery", "delivery_route")
        difficulty: Difficulty level 0.0-1.0
        render: Whether to render visualization
        num_episodes: Number of episodes to run
        max_steps: Maximum steps per episode
        seed: Random seed
    """
    print(f"\n{'='*60}")
    print(f"Drone AI Path Finder Demo")
    print(f"{'='*60}")
    print(f"Task: {task}")
    print(f"Difficulty: {difficulty}")
    print(f"Episodes: {num_episodes}")
    print(f"{'='*60}\n")

    # Create environment
    render_mode = "human" if render else None
    env = make_env(task=task, difficulty=difficulty, render_mode=render_mode, seed=seed)

    for episode in range(num_episodes):
        print(f"\n--- Episode {episode + 1}/{num_episodes} ---")

        obs, info = env.reset(seed=seed + episode if seed else None)
        total_reward = 0
        waypoints_reached = 0

        print(f"Start position: {info['drone_position']}")
        print(f"Goal position: {env.goal_position.tolist()}")
        print(f"Waypoints: {info['total_waypoints']}")

        for step in range(max_steps):
            # Simple reactive controller for demo
            action = compute_demo_action(env, obs)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            # Track waypoints
            if info['current_waypoint_idx'] > waypoints_reached:
                waypoints_reached = info['current_waypoint_idx']
                print(f"  Step {step}: Reached waypoint {waypoints_reached}")

            # Render
            if render:
                env.render()
                time.sleep(0.02)

            # Check termination
            if terminated or truncated:
                break

        # Episode summary
        print(f"\nEpisode {episode + 1} Summary:")
        print(f"  Steps: {info['step_count']}")
        print(f"  Waypoints reached: {waypoints_reached}/{info['total_waypoints']}")
        print(f"  Distance to goal: {info['distance_to_goal']:.2f}m")
        print(f"  Total reward: {total_reward:.2f}")
        print(f"  Crashed: {info['crashed']}")

    env.close()
    print(f"\n{'='*60}")
    print("Demo complete!")
    print(f"{'='*60}\n")


def compute_demo_action(env: DroneEnv, obs: np.ndarray) -> np.ndarray:
    """Compute a simple reactive action for demo.

    This is a basic proportional controller - not trained, just for demonstration.

    Args:
        env: Environment instance
        obs: Current observation

    Returns:
        Action array [thrust, roll_rate, pitch_rate, yaw_rate]
    """
    # Extract relevant observations
    position = obs[0:3]
    velocity = obs[3:6]
    orientation = obs[6:9]
    to_waypoint = obs[9:12]
    distance = obs[12]

    # Target position is current waypoint
    target = position + to_waypoint

    # Altitude control
    altitude_error = target[2] - position[2]
    thrust = 0.0 + 0.5 * altitude_error - 0.3 * velocity[2]
    thrust = np.clip(thrust, -0.5, 0.8)

    # Horizontal control - compute desired heading
    if np.linalg.norm(to_waypoint[:2]) > 0.1:
        desired_yaw = np.arctan2(to_waypoint[1], to_waypoint[0])
    else:
        desired_yaw = orientation[2]

    # Yaw rate control
    yaw_error = desired_yaw - orientation[2]
    # Normalize to [-pi, pi]
    yaw_error = np.arctan2(np.sin(yaw_error), np.cos(yaw_error))
    yaw_rate = np.clip(yaw_error * 2.0, -1, 1)

    # Pitch control (forward/backward)
    forward_speed = 2.0 if distance > 3.0 else 1.0
    forward_error = forward_speed - np.dot(velocity[:2], to_waypoint[:2] / (distance + 0.1))
    pitch_rate = np.clip(-forward_error * 0.3, -0.5, 0.5)

    # Roll control (minimal)
    roll_rate = -orientation[0] * 2.0

    return np.array([thrust, roll_rate, pitch_rate, yaw_rate], dtype=np.float32)


def random_demo(
    task: str = "delivery_route",
    difficulty: float = 0.5,
    render: bool = True,
    seed: Optional[int] = None
):
    """Run demo with random actions (for testing).

    Args:
        task: Task type
        difficulty: Difficulty level
        render: Whether to render
        seed: Random seed
    """
    print("\nRunning random action demo...")

    render_mode = "human" if render else None
    env = make_env(task=task, difficulty=difficulty, render_mode=render_mode, seed=seed)

    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Observation: {obs}")

    for step in range(200):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        if render:
            env.render()
            time.sleep(0.02)

        if terminated or truncated:
            print(f"\nEpisode ended at step {step}")
            print(f"  Reason: {'terminated' if terminated else 'truncated'}")
            print(f"  Crashed: {info['crashed']}")
            break

    env.close()


def test_path_planning():
    """Test path planning functionality."""
    from .simulation import generate_random_obstacles
    from .path_planning import PathPlanner

    print("\nTesting path planning...")

    # Generate obstacles
    obstacles = generate_random_obstacles(
        10,
        bounds=(-15, 15, -15, 15, 0, 10),
        seed=42
    )

    # Create planner
    planner = PathPlanner(obstacles)

    # Test A* path
    start = np.array([0, 0, 5])
    goal = np.array([10, 10, 5])

    print(f"Planning path from {start} to {goal}")

    path_astar = planner.plan_path(start, goal, algorithm="astar")
    print(f"A* path: {len(path_astar)} waypoints")
    for i, wp in enumerate(path_astar):
        print(f"  WP{i}: {wp}")

    # Test RRT path
    path_rrt = planner.plan_path(start, goal, algorithm="rrt")
    print(f"\nRRT path: {len(path_rrt)} waypoints")
    for i, wp in enumerate(path_rrt):
        print(f"  WP{i}: {wp}")

    # Visualize if matplotlib available
    try:
        from .visualization import visualize_path
        visualize_path(path_astar, obstacles, start, goal, "A* Path")
    except Exception as e:
        print(f"Could not visualize: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Drone AI Demo")
    parser.add_argument("--task", type=str, default="delivery_route",
                        choices=["hover", "delivery", "delivery_route"],
                        help="Task type")
    parser.add_argument("--difficulty", type=float, default=0.5,
                        help="Difficulty level (0.0-1.0)")
    parser.add_argument("--render", action="store_true", default=True,
                        help="Enable rendering")
    parser.add_argument("--no-render", action="store_false", dest="render",
                        help="Disable rendering")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Number of episodes")
    parser.add_argument("--steps", type=int, default=500,
                        help="Max steps per episode")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed")
    parser.add_argument("--random", action="store_true",
                        help="Use random actions (for testing)")
    parser.add_argument("--test-planner", action="store_true",
                        help="Test path planning")

    args = parser.parse_args()

    if args.test_planner:
        test_path_planning()
    elif args.random:
        random_demo(args.task, args.difficulty, args.render, args.seed)
    else:
        run_demo(
            task=args.task,
            difficulty=args.difficulty,
            render=args.render,
            num_episodes=args.episodes,
            max_steps=args.steps,
            seed=args.seed
        )


if __name__ == "__main__":
    main()
