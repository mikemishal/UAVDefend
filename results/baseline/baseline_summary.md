# Baseline Policy Evaluation Summary

*Generated: 2026-03-26 11:47:09*

## Overview

This report summarizes Monte Carlo evaluation of the **Greedy Intercept Policy**,
a hand-designed baseline that pursues the estimated enemy position directly.

### Default Configuration Performance

- **Success Rate**: 42.0%
- **Episodes Evaluated**: 100
- **Mean Episode Length**: 19.5 steps
- **Detection Rate**: 100.0%

## Key Findings

### 1. Defender Speed Impact

- **Best performing speed**: v_d = 20 (50% success)
- **Correlation with success**: r = 0.96 (positive)
- **Minimum speed for 50% success**: v_d ≥ 20
- **Success range**: 22% (v_d=10) → 50% (v_d=20)

> **Insight**: Defender speed has a strong positive effect on success. 
> The greedy policy requires significant speed advantage to achieve reliable interception.

### 2. Enemy Speed Impact

- **Best case**: v_e = 8 (57% success)
- **Worst case**: v_e = 14 (40% success)
- **Correlation with success**: r = -0.80 (negative)
- **Performance degradation**: 17 percentage points across range

> **Insight**: Faster enemies significantly reduce interception success. 
> The greedy policy struggles when speed advantage diminishes.

### 3. Detection Radius Impact

- **Minimum radius for reliable detection (≥95%)**: r_det ≥ 8
- **Detection rate at minimum radius**: 78%
- **Correlation with success**: r = 0.95
- **Success range**: 34% → 42%

> **Insight**: Detection radius primarily affects whether detection occurs at all. 
> Once detection is reliable, further increases provide diminishing returns.

### 4. Speed Ratio Analysis (2D Grid)

- **Best configuration**: v_d=12, v_e=8 (62% success)
- **Worst configuration**: v_d=12, v_e=18 (14% success)
- **Minimum speed ratio for 50% success**: v_d/v_e ≥ 1.50
- **Success at equal speeds (v_d = v_e)**: 33%
- **Overall mean success across grid**: 38%

> **Insight**: The greedy baseline requires substantial speed advantage (ratio > 1.5) 
> for reliable interception. This leaves significant room for RL improvement.

## Conclusions

1. **Speed advantage is critical**: The greedy policy is fundamentally limited by
   pursuit dynamics and requires the defender to be significantly faster than the enemy.

2. **Detection is a threshold effect**: Below a certain detection radius, many episodes
   fail due to missed detection. Above this threshold, detection is reliable.

3. **Baseline ceiling ~50-60%**: Even under favorable conditions (2x speed advantage),
   the greedy policy achieves at most 50-60% success, suggesting predictive/planning
   strategies could significantly outperform pure pursuit.

4. **RL opportunity**: The gap between greedy baseline performance and theoretical
   optimal suggests RL policies could learn anticipatory interception strategies.

## Data Files

- `baseline_results.csv` - Per-episode results at default configuration
- `sweep_defender_speed.csv` - Defender speed parameter sweep
- `sweep_enemy_speed.csv` - Enemy speed parameter sweep
- `sweep_detection_radius.csv` - Detection radius parameter sweep
- `sweep_speed_grid.csv` - 2D defender vs enemy speed grid
- `sweep_speed_grid_heatmap.png` - Heatmap visualization
- `sweep_speed_grid_contour.png` - Contour visualization
