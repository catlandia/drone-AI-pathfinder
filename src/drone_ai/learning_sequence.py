"""Curriculum learning sequence for drone navigation.

Implements a staged training approach:
Stage 1: HOVER - Learn basic flight control
Stage 2: DELIVERY - Navigate to single waypoint with obstacles
Stage 3: DELIVERY_ROUTE - Multi-waypoint navigation with path selection

Run with:
    python -m drone_ai.learning_sequence --render --render-freq 5
"""

import argparse
import numpy as np
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .environment import DroneEnv, TaskType, make_env


@dataclass
class StageConfig:
    """Configuration for a training stage."""
    task: TaskType
    name: str
    min_difficulty: float
    max_difficulty: float
    episodes_per_difficulty: int
    success_threshold: float  # Average reward to advance
    difficulty_increment: float


class CurriculumScheduler:
    """Manages curriculum learning progression."""

    def __init__(self):
        self.stages = [
            StageConfig(
                task=TaskType.HOVER,
                name="Stage 1: Hover Control",
                min_difficulty=0.0,
                max_difficulty=0.5,
                episodes_per_difficulty=50,
                success_threshold=20.0,
                difficulty_increment=0.1
            ),
            StageConfig(
                task=TaskType.DELIVERY,
                name="Stage 2: Point Delivery",
                min_difficulty=0.2,
                max_difficulty=0.7,
                episodes_per_difficulty=100,
                success_threshold=30.0,
                difficulty_increment=0.1
            ),
            StageConfig(
                task=TaskType.DELIVERY_ROUTE,
                name="Stage 3: Route Navigation",
                min_difficulty=0.3,
                max_difficulty=1.0,
                episodes_per_difficulty=200,
                success_threshold=40.0,
                difficulty_increment=0.1
            )
        ]

        self.current_stage_idx = 0
        self.current_difficulty = self.stages[0].min_difficulty
        self.episode_rewards: List[float] = []

    @property
    def current_stage(self) -> StageConfig:
        return self.stages[self.current_stage_idx]

    def should_advance_difficulty(self) -> bool:
        """Check if difficulty should increase."""
        if len(self.episode_rewards) < self.current_stage.episodes_per_difficulty:
            return False

        recent = self.episode_rewards[-self.current_stage.episodes_per_difficulty:]
        avg_reward = np.mean(recent)

        return avg_reward >= self.current_stage.success_threshold

    def should_advance_stage(self) -> bool:
        """Check if should move to next stage."""
        return (self.current_difficulty >= self.current_stage.max_difficulty and
                self.should_advance_difficulty())

    def advance(self) -> bool:
        """Try to advance difficulty or stage.

        Returns:
            True if advanced, False otherwise
        """
        if self.should_advance_stage():
            if self.current_stage_idx < len(self.stages) - 1:
                self.current_stage_idx += 1
                self.current_difficulty = self.current_stage.min_difficulty
                self.episode_rewards = []
                return True
        elif self.should_advance_difficulty():
            self.current_difficulty = min(
                self.current_difficulty + self.current_stage.difficulty_increment,
                self.current_stage.max_difficulty
            )
            return True
        return False

    def record_episode(self, reward: float):
        """Record episode reward."""
        self.episode_rewards.append(reward)

    def get_status(self) -> Dict:
        """Get current training status."""
        recent_rewards = self.episode_rewards[-50:] if self.episode_rewards else []
        return {
            "stage": self.current_stage.name,
            "stage_idx": self.current_stage_idx + 1,
            "total_stages": len(self.stages),
            "task": self.current_stage.task.value,
            "difficulty": self.current_difficulty,
            "episodes": len(self.episode_rewards),
            "recent_avg_reward": np.mean(recent_rewards) if recent_rewards else 0.0,
            "success_threshold": self.current_stage.success_threshold,
        }


class SimplePolicy:
    """Simple reactive policy for demonstration.

    In a real implementation, this would be replaced with a neural network
    trained via PPO, SAC, or similar algorithms.
    """

    def __init__(self, noise_scale: float = 0.1):
        self.noise_scale = noise_scale

        # Simple gains for proportional control
        self.altitude_p = 0.5
        self.altitude_d = 0.3
        self.yaw_p = 2.0
        self.forward_p = 0.3
        self.roll_p = 2.0

    def compute_action(self, obs: np.ndarray) -> np.ndarray:
        """Compute action from observation.

        Args:
            obs: 31-dimensional observation

        Returns:
            4-dimensional action
        """
        # Parse observation
        position = obs[0:3]
        velocity = obs[3:6]
        orientation = obs[6:9]
        to_waypoint = obs[9:12]
        distance = obs[12]
        bearing = obs[20]
        obstacle_ahead = obs[27]

        # Altitude control
        altitude_error = to_waypoint[2]
        thrust = self.altitude_p * altitude_error - self.altitude_d * velocity[2]

        # Adjust thrust based on obstacle proximity
        if obstacle_ahead < 0.3:  # Close obstacle ahead
            thrust += 0.3  # Climb

        thrust = np.clip(thrust, -0.5, 0.8)

        # Yaw control - turn toward waypoint
        yaw_rate = np.clip(self.yaw_p * bearing, -1, 1)

        # Pitch control - forward/backward based on alignment and distance
        alignment = np.cos(bearing)
        if alignment > 0.5:  # Roughly facing target
            forward_speed = min(2.0, distance * 0.3)
            current_forward = np.dot(velocity[:2], np.array([np.cos(orientation[2]), np.sin(orientation[2])]))
            pitch_rate = -self.forward_p * (forward_speed - current_forward)
        else:
            pitch_rate = 0.0

        # Avoid obstacles
        if obstacle_ahead < 0.2:
            pitch_rate = 0.3  # Slow down / pull back

        pitch_rate = np.clip(pitch_rate, -0.5, 0.5)

        # Roll stabilization
        roll_rate = -self.roll_p * orientation[0]

        # Add exploration noise
        action = np.array([thrust, roll_rate, pitch_rate, yaw_rate])
        action += np.random.normal(0, self.noise_scale, 4)

        return np.clip(action, -1, 1).astype(np.float32)


def train_episode(
    env: DroneEnv,
    policy: SimplePolicy,
    max_steps: int = 500,
    render: bool = False,
    render_freq: int = 1
) -> Tuple[float, Dict]:
    """Run one training episode.

    Args:
        env: Environment instance
        policy: Policy to use
        max_steps: Maximum steps
        render: Whether to render
        render_freq: Render every N steps

    Returns:
        (total_reward, info)
    """
    obs, info = env.reset()
    total_reward = 0.0

    for step in range(max_steps):
        action = policy.compute_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if render and step % render_freq == 0:
            env.render()
            time.sleep(0.01)

        if terminated or truncated:
            break

    return total_reward, info


def run_learning_sequence(
    render: bool = False,
    render_freq: int = 5,
    max_episodes: int = 1000,
    seed: Optional[int] = None,
    verbose: bool = True
):
    """Run full curriculum learning sequence.

    Args:
        render: Whether to render
        render_freq: Render every N steps
        max_episodes: Maximum total episodes
        seed: Random seed
        verbose: Print progress
    """
    print("\n" + "=" * 70)
    print("DRONE AI CURRICULUM LEARNING SEQUENCE")
    print("=" * 70)
    print("\nStages:")
    print("  1. HOVER       - Basic flight stabilization")
    print("  2. DELIVERY    - Single waypoint navigation")
    print("  3. DELIVERY_ROUTE - Multi-waypoint path following")
    print("=" * 70 + "\n")

    scheduler = CurriculumScheduler()
    policy = SimplePolicy(noise_scale=0.15)

    render_mode = "human" if render else None
    env = make_env(
        task=scheduler.current_stage.task.value,
        difficulty=scheduler.current_difficulty,
        render_mode=render_mode,
        seed=seed
    )

    episode = 0
    stage_start_episode = 0

    try:
        while episode < max_episodes:
            # Run episode
            total_reward, info = train_episode(
                env, policy,
                max_steps=500,
                render=render,
                render_freq=render_freq
            )

            scheduler.record_episode(total_reward)
            episode += 1

            # Print progress
            if verbose and episode % 10 == 0:
                status = scheduler.get_status()
                print(f"Episode {episode:4d} | "
                      f"{status['stage']} | "
                      f"Diff: {status['difficulty']:.2f} | "
                      f"Reward: {total_reward:7.2f} | "
                      f"Avg: {status['recent_avg_reward']:7.2f} | "
                      f"Goal: {info['distance_to_goal']:.1f}m")

            # Check for advancement
            if scheduler.advance():
                status = scheduler.get_status()
                print("\n" + "-" * 50)
                print(f"ADVANCEMENT! Now at {status['stage']}")
                print(f"Difficulty: {status['difficulty']:.2f}")
                print(f"Episodes in previous stage: {episode - stage_start_episode}")
                print("-" * 50 + "\n")

                stage_start_episode = episode

                # Recreate environment for new stage/difficulty
                env.close()
                env = make_env(
                    task=scheduler.current_stage.task.value,
                    difficulty=scheduler.current_difficulty,
                    render_mode=render_mode,
                    seed=seed
                )

            # Check if completed all stages
            if (scheduler.current_stage_idx == len(scheduler.stages) - 1 and
                scheduler.current_difficulty >= scheduler.current_stage.max_difficulty and
                len(scheduler.episode_rewards) > scheduler.current_stage.episodes_per_difficulty):

                final_avg = np.mean(scheduler.episode_rewards[-100:])
                if final_avg >= scheduler.current_stage.success_threshold:
                    print("\n" + "=" * 70)
                    print("CURRICULUM COMPLETE!")
                    print(f"Total episodes: {episode}")
                    print(f"Final average reward: {final_avg:.2f}")
                    print("=" * 70 + "\n")
                    break

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")

    finally:
        env.close()

    # Final summary
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    status = scheduler.get_status()
    print(f"Final Stage: {status['stage']}")
    print(f"Final Difficulty: {status['difficulty']:.2f}")
    print(f"Total Episodes: {episode}")
    print(f"Final Avg Reward: {status['recent_avg_reward']:.2f}")
    print("=" * 70 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Drone AI Curriculum Learning")
    parser.add_argument("--render", action="store_true",
                        help="Enable rendering")
    parser.add_argument("--render-freq", type=int, default=5,
                        help="Render every N steps")
    parser.add_argument("--episodes", type=int, default=1000,
                        help="Maximum episodes")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--quiet", action="store_true",
                        help="Reduce output")

    args = parser.parse_args()

    run_learning_sequence(
        render=args.render,
        render_freq=args.render_freq,
        max_episodes=args.episodes,
        seed=args.seed,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()
