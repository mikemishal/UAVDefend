"""Environment configuration parameters."""

from dataclasses import dataclass


@dataclass
class EnvConfig:
    """Configuration for the UAV Defend environment."""
    
    # Domain: [-L, L]²
    L: float = 50.0  # Half-size of the 2D square domain
    
    # Time parameters
    max_steps: int = 2000  # Tmax: Maximum steps before episode ends
    dt: float = 0.5  # Time step duration (∆t)
    
    # Numerical stability
    eps: float = 1e-8  # ε: Small value for numerical stability
    
    # Soldier movement parameters
    v_s: float = 3.0  # Soldier speed (same as defender)
    
    # Enemy drone parameters (weaving pursuit)
    v_e: float = 3.0  # Enemy speed
    rho: float = 0.85  # AR(1) coefficient for weave bias persistence (lower = faster direction changes)
    sigma_a: float = 0.5  # Weave bias noise standard deviation (higher = more lateral movement)
    sigma_e: float = 0.15  # Heading noise standard deviation
    weave_amplitude: float = 1.5  # Multiplier for lateral weave component
    
    # Defender drone parameters
    v_d: float = 3.0  # Defender speed (same as enemy - requires strategic interception)
    
    # RL reward shaping parameters
    reward_intercept: float = 100.0  # Reward for intercepting enemy (WIN)
    reward_soldier_caught: float = -100.0  # Penalty for enemy catching soldier (LOSS)
    reward_collision: float = -100.0  # Penalty for collision loss
    reward_timeout: float = -100.0  # Penalty for timeout (failed to intercept)
    reward_progress_scale: float = 5.0  # Scale for distance progress reward
    reward_time_penalty: float = -0.05  # Small time penalty per step
    
    # Distance thresholds
    detection_radius: float = 8.0  # Defender detects enemy within this radius (limited sensing range)
    intercept_radius: float = 2.0  # Defender intercepts enemy (WIN)
    threat_radius: float = 3.0  # Enemy catches soldier (LOSS)
    collision_radius: float = 5.0  # If enemy & defender both near soldier (LOSS)
