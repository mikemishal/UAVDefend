"""
PPO Policy Wrapper - Wrapper for trained Stable-Baselines3 PPO models.

Provides a unified interface for evaluating trained RL policies
using the same evaluation infrastructure as baseline policies.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class PPOPolicyWrapper:
    """
    Wrapper for Stable-Baselines3 PPO models.
    
    Provides the standard `act(obs, info)` interface for compatibility
    with the evaluation utilities.
    
    Attributes:
        model: The loaded SB3 PPO model.
        deterministic: If True, use deterministic actions (no exploration).
    
    Example:
        >>> policy = PPOPolicyWrapper.load("models/policies/ppo_defender.zip")
        >>> obs, info = env.reset()
        >>> action = policy.act(obs, info)
    """
    
    def __init__(self, model: Any, deterministic: bool = True):
        """
        Initialize the PPO policy wrapper.
        
        Args:
            model: A loaded Stable-Baselines3 PPO model.
            deterministic: If True, use deterministic actions during evaluation.
        """
        self.model = model
        self.deterministic = deterministic
    
    @classmethod
    def load(cls, path: str | Path, deterministic: bool = True) -> "PPOPolicyWrapper":
        """
        Load a PPO model from a saved checkpoint.
        
        Args:
            path: Path to the saved model (.zip file).
            deterministic: If True, use deterministic actions.
        
        Returns:
            PPOPolicyWrapper instance with the loaded model.
        
        Raises:
            ImportError: If stable-baselines3 is not installed.
            FileNotFoundError: If the model file doesn't exist.
        """
        try:
            from stable_baselines3 import PPO
        except ImportError:
            raise ImportError(
                "stable-baselines3 is required for PPOPolicyWrapper. "
                "Install with: pip install stable-baselines3"
            )
        
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        
        model = PPO.load(str(path))
        return cls(model, deterministic=deterministic)
    
    def act(self, obs: np.ndarray, info: dict) -> np.ndarray:
        """
        Get action from the PPO model.
        
        Args:
            obs: Environment observation array.
            info: Environment info dict (unused by PPO).
        
        Returns:
            action: 2D action vector from the policy.
        """
        action, _ = self.model.predict(obs, deterministic=self.deterministic)
        return np.asarray(action, dtype=np.float32)
    
    def reset(self) -> None:
        """Reset internal state (no-op for PPO)."""
        pass
    
    def __repr__(self) -> str:
        return f"PPOPolicyWrapper(deterministic={self.deterministic})"
