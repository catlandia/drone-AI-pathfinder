# Drone AI Path Finder - Makefile
#
# Quick commands:
#   make install      - Install package
#   make demo         - Run demo
#   make train        - Train AI
#   make help         - Show all commands

.PHONY: help install install-full demo train test clean

# Default target
help:
	@echo ""
	@echo "Drone AI Path Finder"
	@echo "===================="
	@echo ""
	@echo "Installation:"
	@echo "  make install       Install core package"
	@echo "  make install-viz   Install with visualization"
	@echo "  make install-full  Install with AI training (stable-baselines3)"
	@echo ""
	@echo "Running:"
	@echo "  make demo          Run demo with math controller"
	@echo "  make demo-render   Run demo with visualization"
	@echo "  make train         Train AI with PPO (requires install-full)"
	@echo "  make curriculum    Run curriculum learning"
	@echo ""
	@echo "Development:"
	@echo "  make test          Run tests"
	@echo "  make clean         Remove build artifacts"
	@echo ""

# Installation targets
install:
	pip install -e .

install-viz:
	pip install -e ".[viz]"

install-full:
	pip install -e ".[all]"
	pip install stable-baselines3

# Running targets
demo:
	python -m drone_ai.demo --task delivery_route --difficulty 0.5

demo-render:
	python -m drone_ai.demo --task delivery_route --difficulty 0.5 --render

demo-hover:
	python -m drone_ai.demo --task hover --render

demo-hard:
	python -m drone_ai.demo --task delivery_route --difficulty 0.9 --render

train:
	python -m drone_ai.train_rl --algo ppo --task hover --timesteps 50000

train-delivery:
	python -m drone_ai.train_rl --algo ppo --task delivery --timesteps 100000

train-route:
	python -m drone_ai.train_rl --algo ppo --task delivery_route --timesteps 200000

curriculum:
	python -m drone_ai.learning_sequence

curriculum-render:
	python -m drone_ai.learning_sequence --render --render-freq 5

# Test path planning
plan:
	python -m drone_ai.demo --test-planner

# Development targets
test:
	PYTHONPATH=src pytest tests/ -v

test-quick:
	PYTHONPATH=src pytest tests/ -v -x

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf src/*.egg-info/
	rm -rf __pycache__/
	rm -rf src/drone_ai/__pycache__/
	rm -rf tests/__pycache__/
	rm -rf .pytest_cache/
	rm -rf models/
