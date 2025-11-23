#!/usr/bin/env python3
"""Train drone AI with Stable-Baselines3.

This script shows how to train a real neural network policy using
reinforcement learning (PPO algorithm).

Install first:
    pip install stable-baselines3

Run:
    python examples/train_with_sb3.py
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("=" * 60)
    print("Stable-Baselines3 not installed!")
    print("Install with: pip install stable-baselines3")
    print("=" * 60)

from drone_ai.environment import DroneEnv, TaskType


def make_drone_env(task="hover", difficulty=0.3, seed=None):
    """Factory function for creating drone environments."""
    def _init():
        env = DroneEnv(
            task=TaskType(task),
            difficulty=difficulty,
            seed=seed
        )
        return env
    return _init


def train_ppo(
    task: str = "hover",
    difficulty: float = 0.3,
    total_timesteps: int = 100_000,
    n_envs: int = 4,
    save_path: str = "models/drone_ppo"
):
    """Train using PPO algorithm.

    Args:
        task: Task type (hover, delivery, delivery_route)
        difficulty: Difficulty level 0.0-1.0
        total_timesteps: Total training steps
        n_envs: Number of parallel environments
        save_path: Where to save the model
    """
    print(f"\n{'='*60}")
    print(f"Training PPO on {task} task (difficulty={difficulty})")
    print(f"{'='*60}\n")

    # Create vectorized environment (parallel training)
    env = make_vec_env(
        make_drone_env(task, difficulty),
        n_envs=n_envs
    )

    # Create evaluation environment
    eval_env = DroneEnv(task=TaskType(task), difficulty=difficulty)

    # Callbacks
    os.makedirs(save_path, exist_ok=True)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_path,
        log_path=save_path,
        eval_freq=5000,
        deterministic=True
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=save_path,
        name_prefix="drone_ppo"
    )

    # Create PPO model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        tensorboard_log=f"{save_path}/tensorboard/"
    )

    # Train!
    print("Starting training...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True
    )

    # Save final model
    model.save(f"{save_path}/drone_ppo_final")
    print(f"\nModel saved to {save_path}/drone_ppo_final")

    env.close()
    eval_env.close()

    return model


def train_curriculum(total_timesteps_per_stage: int = 50_000):
    """Train using curriculum learning through all stages.

    Progresses: HOVER -> DELIVERY -> DELIVERY_ROUTE
    """
    print("\n" + "=" * 60)
    print("CURRICULUM LEARNING WITH PPO")
    print("=" * 60 + "\n")

    stages = [
        ("hover", 0.3),
        ("hover", 0.5),
        ("delivery", 0.3),
        ("delivery", 0.5),
        ("delivery_route", 0.3),
        ("delivery_route", 0.5),
        ("delivery_route", 0.7),
    ]

    model = None

    for i, (task, difficulty) in enumerate(stages):
        print(f"\n--- Stage {i+1}/{len(stages)}: {task} (diff={difficulty}) ---\n")

        env = make_vec_env(make_drone_env(task, difficulty), n_envs=4)

        if model is None:
            # First stage - create new model
            model = PPO("MlpPolicy", env, verbose=1)
        else:
            # Continue from previous model
            model.set_env(env)

        model.learn(
            total_timesteps=total_timesteps_per_stage,
            progress_bar=True,
            reset_num_timesteps=False
        )

        # Save checkpoint
        model.save(f"models/curriculum/stage_{i+1}_{task}")
        env.close()

    print("\n" + "=" * 60)
    print("Curriculum training complete!")
    print("Final model: models/curriculum/stage_7_delivery_route")
    print("=" * 60 + "\n")

    return model


def evaluate_model(model_path: str, task: str = "delivery_route", episodes: int = 5):
    """Evaluate a trained model.

    Args:
        model_path: Path to saved model
        task: Task to evaluate on
        episodes: Number of evaluation episodes
    """
    print(f"\nEvaluating {model_path} on {task}...")

    model = PPO.load(model_path)
    env = DroneEnv(task=TaskType(task), difficulty=0.5, render_mode="human")

    for ep in range(episodes):
        obs, info = env.reset()
        total_reward = 0
        steps = 0

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

            env.render()

            if terminated or truncated:
                break

        print(f"Episode {ep+1}: reward={total_reward:.1f}, steps={steps}")

    env.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train Drone AI with RL")
    parser.add_argument("--algo", choices=["ppo", "curriculum"], default="ppo")
    parser.add_argument("--task", default="hover",
                        choices=["hover", "delivery", "delivery_route"])
    parser.add_argument("--difficulty", type=float, default=0.3)
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--eval", type=str, help="Path to model to evaluate")

    args = parser.parse_args()

    if not SB3_AVAILABLE:
        print("\nPlease install stable-baselines3:")
        print("  pip install stable-baselines3")
        return

    if args.eval:
        evaluate_model(args.eval, args.task)
    elif args.algo == "curriculum":
        train_curriculum(args.timesteps)
    else:
        train_ppo(args.task, args.difficulty, args.timesteps)


if __name__ == "__main__":
    main()
