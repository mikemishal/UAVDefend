# uav_defend/policies - Defender decision logic
#
# This module provides both baseline (scripted) and RL (trained) policies
# with a unified interface for evaluation.
#
# All policies implement:
#   - act(obs: np.ndarray, info: dict) -> np.ndarray
#   - reset() -> None
#
# This allows the same evaluation code to work for both baseline and RL.

# Baseline policies
from uav_defend.policies.baseline import GreedyInterceptPolicy, RandomPolicy

# RL policy wrappers
from uav_defend.policies.rl import PPOPolicyWrapper

# Policy registry (factory pattern)
from uav_defend.policies.registry import get_policy, list_policies, get_policy_name

__all__ = [
    # Baseline policies
    "GreedyInterceptPolicy",
    "RandomPolicy",
    # RL wrapper
    "PPOPolicyWrapper",
    # Registry
    "get_policy",
    "list_policies",
    "get_policy_name",
]
