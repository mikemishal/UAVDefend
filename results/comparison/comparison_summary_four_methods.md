# Four-Method Comparison Report

## UAV Defender: Baseline vs Kalman Baseline vs Direct RL vs RL-Kalman

**Generated:** 2026-04-02 22:45

---

## Overall Performance Comparison

| Method | Episodes | Success Rate | 95% CI | Failure Rate | Timeout Rate | Mean Episode Length | Mean Tracking Error |
|--------|----------|--------------|--------|--------------|--------------|---------------------|---------------------|
| Greedy Baseline | 1000 | 42.5% ± 1.6% | [39.5%, 45.6%] | 57.5% | 0.0% | 19.0 | N/A |
| Kalman Baseline | 1000 | 42.9% ± 1.6% | [39.9%, 46.0%] | 57.1% | 0.0% | 19.2 | 1.9048 |
| PPO (Direct RL) | 1000 | 60.3% ± 1.5% | [57.2%, 63.3%] | 39.7% | 0.0% | 16.1 | N/A |
| PPO (RL-Kalman) | 1000 | 57.7% ± 1.6% | [54.6%, 60.7%] | 42.3% | 0.0% | 16.6 | 1.6089 |

## Advisor Questions and Answers

### 1. Does Kalman improve the greedy baseline?

Kalman Baseline vs Greedy Baseline: 0.4% (p=0.8565, n.s.).
Result: No statistically significant controller-independent gain from Kalman filtering.

### 2. Does RL improve over both baselines?

Direct RL vs Greedy Baseline: 17.8% (p=0.0000, ***).
Direct RL vs Kalman Baseline: 17.4% (p=0.0000, ***).
Result: Learned control outperforms both hand-designed baselines.

### 3. Does RL-Kalman improve over Direct RL?

RL-Kalman vs Direct RL: -2.6% (p=0.2372, n.s.).
Result: RL-Kalman and Direct RL are statistically similar at current sample size.

### 4. Which contributes more: estimation or control?

Estimation-only gain (Greedy → Kalman Baseline): 0.4%
Control-only gain (Greedy → Direct RL): 17.8%
Combined gain (Greedy → RL-Kalman): 15.2%
Within-RL estimation effect (Direct RL → RL-Kalman): -2.6%

Interpretation: Performance gains are dominated by the control component (learning), not by estimation alone.

---

## Parameter-Sweep Comparison (Method Ranking)

### Defender speed sweep

- Greedy Baseline: 0/6 best settings
- Kalman Baseline: 0/6 best settings
- PPO (Direct RL): 3/6 best settings
- PPO (RL-Kalman): 3/6 best settings

### Enemy speed sweep

- Greedy Baseline: 0/5 best settings
- Kalman Baseline: 1/5 best settings
- PPO (Direct RL): 1/5 best settings
- PPO (RL-Kalman): 3/5 best settings

### Detection radius sweep

- Greedy Baseline: 0/5 best settings
- Kalman Baseline: 0/5 best settings
- PPO (Direct RL): 0/5 best settings
- PPO (RL-Kalman): 5/5 best settings

### Speed-grid sweep

Mean success rate across all defender/enemy speed grid points:

- Greedy Baseline: 38.0%
- Kalman Baseline: 36.3%
- PPO (Direct RL): 50.2%
- PPO (RL-Kalman): 45.3%

---

## Research Conclusion

Kalman baseline is numerically better, but the gain is not statistically significant.

Across methods, the dominant source of performance gain is the controller design (learned policy), while estimation contributes a smaller and context-dependent effect.

---

*Report generated automatically by `experiments/comparison/report_all_methods.py`*
