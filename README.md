# UAV Defend

A 2D discrete-time simulator for drone defense using Reinforcement Learning.

## Overview

This project implements a Gymnasium-compatible environment where:
- A **soldier** performs a random walk within a bounded domain
- An **enemy drone** pursues the soldier with a weaving attack pattern
- A **defender drone** must intercept the enemy before it reaches the soldier

The defender is trained using PPO (Proximal Policy Optimization) from Stable-Baselines3.

## Environment

- **Domain:** Ω = [-50, 50]²
- **Speeds:** v_defender = v_enemy = 3.0 (equal speeds make the problem non-trivial)
- **Intercept radius:** 2.0 units
- **Threat radius:** 3.0 units

### Observation Space
8-dimensional normalized vector in [-1, 1]:
- Soldier position (x, y)
- Defender position (x, y)
- Enemy position (x, y)
- Defender-to-enemy direction vector (x, y)

### Action Space
2-dimensional continuous vector in [-1, 1] representing the defender's heading direction.

### Reward Structure
- **Progress reward:** `5.0 * (prev_dist - curr_dist) - 0.05` per step
- **Intercept (win):** +100
- **Soldier caught / collision / timeout (loss):** -100

## Installation

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Train PPO Agent
```bash
python train_ppo.py
```
This trains for 200k timesteps and saves the model as `ppo_defender.zip`.

### Evaluate & Generate Video
```bash
python evaluate_ppo.py
```
This runs evaluation episodes and creates `ppo_evaluation.mp4`.

### Visualize Environment
```bash
python visualize_soldier.py
```
Creates animations of episodes with random or scripted policies.

## Project Structure

```
UAV_Defend/
├── uav_defend/
│   ├── __init__.py
│   ├── config.py          # Environment configuration
│   └── soldier_env.py     # Main Gymnasium environment
├── train_ppo.py           # PPO training script
├── evaluate_ppo.py        # Model evaluation & video generation
├── visualize_soldier.py   # Visualization utilities
├── requirements.txt
└── README.md
```

## Results

With 200k training timesteps:
- **Win rate:** ~60%
- **Average reward:** ~290

## Dependencies

- gymnasium
- numpy
- matplotlib
- stable-baselines3
- torch
- opencv-python

## License

MIT
