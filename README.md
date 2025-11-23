# Drone AI Path Finder

AI-powered drone path selection and navigation using reinforcement learning.

## Architecture

This implements **Layer 2 (Path Finder AI)** of a 4-layer drone AI system:

```
┌─────────────────────────────────────┐
│  Layer 1: Manager AI                │  High-level mission planning
├─────────────────────────────────────┤
│  Layer 2: Path Finder AI  ◄── THIS  │  Path selection & obstacle avoidance
├─────────────────────────────────────┤
│  Layer 3: Perception                │  Sensing & environment mapping
├─────────────────────────────────────┤
│  Layer 4: Flying                    │  Low-level motor control
└─────────────────────────────────────┘
```

## Quick Install

### Option 1: One-line installer (easiest)

```bash
git clone <repo-url>
cd drone-AI-pathfinder
./install.sh              # Basic install
./install.sh --full       # With AI training support
./install.sh --help       # See all options
```

### Option 2: Using Make

```bash
git clone <repo-url>
cd drone-AI-pathfinder
make install              # Basic install
make install-full         # With AI training
make help                 # See all commands
```

### Option 3: Manual pip install

```bash
git clone <repo-url>
cd drone-AI-pathfinder
pip install -e .          # Basic
pip install -e ".[viz]"   # With visualization
pip install -e ".[all]"   # Everything
```

## Quick Start

### 1. Run the Demo (Math-based controller)

```bash
# Basic demo - uses simple proportional controller (no AI)
python -m drone_ai.demo --task delivery_route --difficulty 0.5

# With visualization (requires matplotlib or pygame)
python -m drone_ai.demo --task delivery_route --difficulty 0.7 --render
```

### 2. Run Curriculum Learning (Math-based)

```bash
# Runs through HOVER → DELIVERY → DELIVERY_ROUTE stages
python -m drone_ai.learning_sequence --render
```

### 3. Train with Real AI (Reinforcement Learning)

```bash
# Install RL library
pip install stable-baselines3

# Train with PPO
python -m drone_ai.train_rl --algo ppo --timesteps 100000

# Or run the quick example:
python examples/train_with_sb3.py
```

## What's AI vs What's Math?

| Component | Type | Description |
|-----------|------|-------------|
| A* Path Planning | **Math** | Graph search algorithm |
| RRT Path Planning | **Math** | Randomized tree search |
| Drone Physics | **Math** | Newtonian mechanics simulation |
| Demo Controller | **Math** | Proportional-derivative control |
| **RL Policy** | **AI** | Neural network trained via PPO/SAC |

The environment is a **Gymnasium RL environment** - it's designed to train AI agents, but the demo uses a simple math-based controller so you can see it work immediately.

## Project Structure

```
drone-AI-pathfinder/
├── src/drone_ai/
│   ├── simulation.py      # Drone physics, Obstacle class
│   ├── environment.py     # Gymnasium RL environment (31-dim obs)
│   ├── path_planning.py   # A* and RRT algorithms
│   ├── visualization.py   # 2D/3D rendering
│   ├── demo.py            # Demo with math controller
│   └── learning_sequence.py  # Curriculum learning
├── examples/
│   └── train_with_sb3.py  # Train with Stable-Baselines3
├── tests/
└── pyproject.toml
```

## Training Stages (Curriculum Learning)

1. **HOVER** - Learn to stabilize at a point
2. **DELIVERY** - Navigate to single waypoint with obstacles
3. **DELIVERY_ROUTE** - Multi-waypoint path following

## Observation Space (31 dimensions)

| Index | Feature |
|-------|---------|
| 0-2 | Drone position (x, y, z) |
| 3-5 | Drone velocity |
| 6-8 | Drone orientation (roll, pitch, yaw) |
| 9-11 | Vector to current waypoint |
| 12 | Distance to current waypoint |
| 13-16 | Vector/distance to next waypoint |
| 17-19 | Vector to waypoint after next |
| 20 | Bearing angle to waypoint |
| 21 | Path curvature ahead |
| 22 | Remaining waypoints (normalized) |
| 23 | Nearest obstacle distance |
| 24-26 | Direction to nearest obstacle |
| 27 | Obstacle proximity ahead |
| 28-30 | Goal position |

## Action Space (4 dimensions)

| Index | Action | Range |
|-------|--------|-------|
| 0 | Thrust | [-1, 1] |
| 1 | Roll rate | [-1, 1] |
| 2 | Pitch rate | [-1, 1] |
| 3 | Yaw rate | [-1, 1] |

## Example: Custom Environment Usage

```python
from drone_ai import DroneEnv, PathPlanner
import numpy as np

# Create environment
env = DroneEnv(task="delivery_route", difficulty=0.7)
obs, info = env.reset()

print(f"Waypoints to visit: {len(env.route_waypoints)}")
print(f"Obstacles: {len(env.obstacles)}")

# Run episode
for step in range(500):
    # Your policy here (or random)
    action = env.action_space.sample()

    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        break

env.close()
```

## Example: Use Path Planner Directly

```python
from drone_ai import PathPlanner, Obstacle
import numpy as np

# Create obstacles
obstacles = [
    Obstacle(position=[5, 5, 5], size=[2.0]),  # Sphere
]

# Create planner
planner = PathPlanner(obstacles)

# Plan path
start = np.array([0, 0, 5])
goal = np.array([10, 10, 5])

path = planner.plan_path(start, goal, algorithm="astar")
print(f"Path: {len(path)} waypoints")

# Check if direct path is clear
if planner.is_path_clear(start, goal):
    print("Direct path available!")
```

## Requirements

- Python 3.8+
- numpy
- gymnasium

Optional:
- matplotlib (3D visualization)
- pygame (2D visualization)
- stable-baselines3 (AI training)
