"""Reinforcement learning training module.

Provides a simple interface for training with Stable-Baselines3.

Usage:
    python -m drone_ai.train_rl --algo ppo --task hover --timesteps 50000
"""

import argparse
import os
import sys


def check_sb3():
    """Check if Stable-Baselines3 is available."""
    try:
        import stable_baselines3
        return True
    except ImportError:
        return False


def train(
    algo: str = "ppo",
    task: str = "hover",
    difficulty: float = 0.3,
    timesteps: int = 100_000,
    save_dir: str = "models"
):
    """Train a reinforcement learning agent.

    Args:
        algo: Algorithm to use (ppo or sac)
        task: Task type
        difficulty: Difficulty level
        timesteps: Total training timesteps
        save_dir: Directory to save models
    """
    if not check_sb3():
        print("=" * 60)
        print("Stable-Baselines3 is required for RL training.")
        print("")
        print("Install with:")
        print("  pip install stable-baselines3")
        print("")
        print("Or run the math-based demo instead:")
        print("  python -m drone_ai.demo --task delivery_route")
        print("=" * 60)
        return

    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.env_util import make_vec_env

    from .environment import DroneEnv, TaskType

    print(f"\n{'='*60}")
    print(f"Training {algo.upper()} on '{task}' task")
    print(f"Difficulty: {difficulty}, Timesteps: {timesteps:,}")
    print(f"{'='*60}\n")

    # Create environment factory
    def make_env():
        return DroneEnv(task=TaskType(task), difficulty=difficulty)

    # Create vectorized environment
    env = make_vec_env(make_env, n_envs=4)

    # Create model
    if algo.lower() == "sac":
        model = SAC("MlpPolicy", env, verbose=1)
    else:
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
        )

    # Train
    model.learn(total_timesteps=timesteps, progress_bar=True)

    # Save
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/drone_{algo}_{task}"
    model.save(save_path)
    print(f"\nModel saved to: {save_path}")

    env.close()
    return model


def main():
    parser = argparse.ArgumentParser(description="Train Drone AI with RL")
    parser.add_argument("--algo", choices=["ppo", "sac"], default="ppo",
                        help="RL algorithm")
    parser.add_argument("--task", default="hover",
                        choices=["hover", "delivery", "delivery_route"])
    parser.add_argument("--difficulty", type=float, default=0.3)
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--save-dir", default="models")

    args = parser.parse_args()

    train(
        algo=args.algo,
        task=args.task,
        difficulty=args.difficulty,
        timesteps=args.timesteps,
        save_dir=args.save_dir
    )


if __name__ == "__main__":
    main()
