"""
Greedy Intercept Policy - Baseline defender policy

A simple hand-designed policy that greedily moves the defender
toward the predicted intercept point of the attacker.
"""


class GreedyInterceptPolicy:
    """
    Greedy baseline policy for UAV defense.
    
    Moves the defender directly toward the attacker's predicted position,
    attempting to intercept before the attacker reaches the target.
    """
    
    def __init__(self):
        pass
    
    def get_action(self, observation):
        """
        Compute the greedy intercept action based on observation.
        
        Args:
            observation: Environment observation containing defender and attacker state.
            
        Returns:
            action: The action to take.
        """
        raise NotImplementedError("Implement greedy intercept logic")
    
    def reset(self):
        """Reset any internal state."""
        pass
