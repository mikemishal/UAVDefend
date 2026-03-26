# RL Training Results

This directory contains trained RL models and checkpoints.

## Files

- `ppo_defender_final.zip` - Final trained PPO model
- `ppo_checkpoint_*.zip` - Periodic checkpoints during training
- `best_model.zip` - Best model based on evaluation reward

## Usage

Evaluate a trained model:
```bash
python experiments/evaluate_policy.py --policy ppo \
    --model-path results/rl/models/ppo_defender_final.zip \
    --n-episodes 1000
```
