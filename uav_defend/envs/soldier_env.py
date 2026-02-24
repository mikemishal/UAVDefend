"""Minimal Gymnasium environment with a stochastically moving soldier."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from uav_defend.config.env_config import EnvConfig


class SoldierEnv(gym.Env):
    """
    A 2D discrete-time environment for RL training.
    
    The defender drone is controlled by the RL agent to intercept
    an enemy drone before it reaches the soldier.
    
    Domain: Ω = [-L, L]² (square centered at origin).
    
    Entities:
        - Soldier (s ∈ ℝ²): Gaussian random walk (uncontrolled)
        - Defender (d ∈ ℝ²): Controlled by RL agent via 2D heading action
        - Enemy (e ∈ ℝ²): Weaving pursuit toward soldier (uncontrolled)
    
    Observation (shape=8, normalized to [-1, 1]):
        [soldier_x, soldier_y, defender_x, defender_y, enemy_x, enemy_y,
         defender_to_enemy_x, defender_to_enemy_y]  # Relative vector
        All positions normalized by L.
    
    Action (shape=2):
        2D heading vector in [-1, 1]². Normalized to unit vector.
        Defender moves: d_next = d + v_d * dt * normalized_action
    
    Reward (dense shaping for RL):
        +100 for intercepting enemy (WIN)
        -100 for soldier caught (LOSS)
        -100 for collision loss
        -50 for timeout
        +0.1 * (prev_dist - curr_dist) for closing distance to enemy
        -0.1 per step (encourages efficiency)
    
    Termination:
        - Intercepted: defender_enemy_dist < intercept_radius (WIN)
        - Soldier caught: enemy_soldier_dist < threat_radius (LOSS)
        - Collision: both within collision_radius of soldier (LOSS)
        - Timeout: step_count >= max_steps
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(
        self,
        config: EnvConfig | None = None,
        render_mode: str | None = None,
    ):
        """
        Initialize the SoldierEnv.
        
        Args:
            config: Environment configuration. Uses defaults if None.
            render_mode: One of "human", "rgb_array", or None.
        """
        super().__init__()
        
        self.config = config if config is not None else EnvConfig()
        self.render_mode = render_mode
        
        # Observation space: normalized positions + relative vectors
        # [soldier, defender, enemy, defender_to_enemy] all in [-1, 1]
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(8,),
            dtype=np.float32,
        )
        
        # Track previous distance for reward shaping
        self._prev_defender_enemy_dist: float = 0.0
        
        # Action space: 2D heading vector for defender
        # Agent outputs a direction in [-1, 1]², normalized to unit vector
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )
        
        # Internal state
        self._soldier_pos: np.ndarray | None = None
        self._defender_pos: np.ndarray | None = None
        self._enemy_pos: np.ndarray | None = None
        self._weave_bias: float = 0.0  # AR(1) lateral weave bias 'a'
        self._step_count: int = 0
        self._np_random: np.random.Generator | None = None
    
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        """
        Reset the environment to an initial state.
        
        The soldier is initialized at the origin (0, 0).
        
        Args:
            seed: Random seed for reproducibility.
            options: Additional options (unused).
        
        Returns:
            observation: Soldier position with shape (2,).
            info: Additional information dict.
        """
        super().reset(seed=seed)
        self._np_random = np.random.default_rng(seed)
        
        # Initialize soldier at the origin
        self._soldier_pos = np.array([0.0, 0.0], dtype=np.float32)
        
        # Initialize defender co-located with soldier: d0 = s0
        self._defender_pos = self._soldier_pos.copy()
        
        # Initialize enemy at random position on boundary edge
        self._enemy_pos = self._spawn_enemy_at_edge()
        
        # Initialize weave bias to zero: a0 = 0
        self._weave_bias = 0.0
        
        self._step_count = 0
        
        # Calculate initial distances
        initial_enemy_soldier_dist = np.linalg.norm(self._enemy_pos - self._soldier_pos)
        initial_defender_enemy_dist = np.linalg.norm(self._defender_pos - self._enemy_pos)
        
        # Store for reward shaping
        self._prev_defender_enemy_dist = initial_defender_enemy_dist
        
        return self._get_obs(), self._get_info("ongoing", initial_enemy_soldier_dist, initial_defender_enemy_dist)
    
    def _spawn_enemy_at_edge(self) -> np.ndarray:
        """
        Spawn enemy at a random location on the boundary of [-L, L]².
        
        Uniformly choose one of four edges, then sample a coordinate
        uniformly along that edge.
        
        Returns:
            Enemy position array of shape (2,).
        """
        L = self.config.L
        edge = self._np_random.integers(0, 4)  # 0=top, 1=bottom, 2=left, 3=right
        coord = self._np_random.uniform(-L, L)  # Position along the edge
        
        if edge == 0:  # Top edge: y = L
            return np.array([coord, L], dtype=np.float32)
        elif edge == 1:  # Bottom edge: y = -L
            return np.array([coord, -L], dtype=np.float32)
        elif edge == 2:  # Left edge: x = -L
            return np.array([-L, coord], dtype=np.float32)
        else:  # Right edge: x = L
            return np.array([L, coord], dtype=np.float32)
    
    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one time step in the environment.
        
        Args:
            action: 2D heading vector for defender in [-1, 1]².
                   Will be normalized to unit vector.
        
        Returns:
            observation: Concatenated positions [soldier, defender, enemy].
            reward: +100 (intercept), -100 (loss), 0 (ongoing).
            terminated: True if terminal condition reached.
            truncated: False.
            info: Additional information dict.
        """
        assert self._soldier_pos is not None, "Call reset() before step()"
        
        # Move soldier stochastically
        self._move_soldier()
        
        # Move enemy with weaving pursuit toward soldier
        self._move_enemy()
        
        # Move defender based on agent's action (2D heading)
        self._move_defender(action)
        
        self._step_count += 1
        
        # Calculate distances
        enemy_soldier_dist = np.linalg.norm(self._enemy_pos - self._soldier_pos)
        defender_enemy_dist = np.linalg.norm(self._defender_pos - self._enemy_pos)
        defender_soldier_dist = np.linalg.norm(self._defender_pos - self._soldier_pos)
        
        # Check termination conditions
        # WIN: Defender intercepts enemy
        intercepted = defender_enemy_dist < self.config.intercept_radius
        
        # LOSS: Enemy reaches soldier
        soldier_caught = enemy_soldier_dist < self.config.threat_radius
        
        # LOSS: Enemy and defender both within collision radius of soldier
        # (defender failed to intercept before enemy reached soldier area)
        collision_loss = (enemy_soldier_dist < self.config.collision_radius and 
                         defender_soldier_dist < self.config.collision_radius and
                         not intercepted)
        
        # Episode terminates on any terminal condition or timeout
        terminated = intercepted or soldier_caught or collision_loss or (self._step_count >= self.config.max_steps)
        truncated = False
        
        # Determine outcome and reward
        # dist_de = defender to enemy, dist_es = enemy to soldier
        dist_de = defender_enemy_dist
        dist_es = enemy_soldier_dist
        
        if intercepted:
            outcome = "intercepted"
            reward = 100.0  # WIN
        elif soldier_caught:
            outcome = "soldier_caught"
            reward = -100.0  # LOSS
        elif collision_loss:
            outcome = "collision_loss"
            reward = -100.0  # LOSS
        elif self._step_count >= self.config.max_steps:
            outcome = "timeout"
            reward = -100.0  # LOSS (failed to intercept)
        else:
            outcome = "ongoing"
            # Dense reward shaping:
            # 1. Progress reward: positive for closing distance to enemy
            progress = self._prev_defender_enemy_dist - dist_de
            reward = 5.0 * progress
            # 2. Small time penalty
            reward -= 0.05
        
        # Update previous distance for next step
        self._prev_defender_enemy_dist = defender_enemy_dist
        
        return self._get_obs(), reward, terminated, truncated, self._get_info(
            outcome, enemy_soldier_dist, defender_enemy_dist
        )
    
    def _move_defender(self, action: np.ndarray) -> None:
        """
        Move the defender drone based on agent's action.
        
        Args:
            action: 2D heading vector from RL agent in [-1, 1]².
        
        Behavior:
        - Normalize action to unit vector (if norm > eps)
        - Move defender: d_next = d + v_d * dt * normalized_action
        - Apply reflecting boundary conditions
        - If action norm < eps: defender does not move
        """
        eps = self.config.eps
        
        # Ensure action is numpy array
        action = np.asarray(action, dtype=np.float32)
        
        # Calculate action norm
        action_norm = np.linalg.norm(action)
        
        # If action is too small, defender doesn't move
        if action_norm < eps:
            return
        
        # Normalize action to unit vector
        direction = action / action_norm
        
        # Move defender
        displacement = self.config.v_d * self.config.dt * direction
        new_pos = self._defender_pos + displacement
        
        # Apply reflecting boundary conditions
        new_pos = self._reflect_boundary(new_pos)
        self._defender_pos = new_pos.astype(np.float32)
    
    def _move_soldier(self) -> None:
        """
        Move the soldier with a true Gaussian random walk.
        
        - Sample 2D Gaussian displacement with scale σ = v_s * dt
        - Variable step magnitude (Gaussian-distributed)
        - Reflecting boundary conditions at [-L, L]²
        """
        # True Gaussian random walk: sample displacement directly
        # Scale (standard deviation) determines typical step size
        sigma = self.config.v_s * self.config.dt
        displacement = self._np_random.normal(loc=0.0, scale=sigma, size=(2,))
        
        # Update position
        new_pos = self._soldier_pos + displacement
        
        # Reflecting boundary conditions at [-L, L]²
        new_pos = self._reflect_boundary(new_pos)
        
        self._soldier_pos = new_pos.astype(np.float32)
    
    def _move_enemy(self) -> None:
        """
        Move the enemy with a stochastic weaving pursuit policy.
        
        Weaving pursuit dynamics:
        1. Compute vector to soldier: r = s - e
        2. Unit vector toward soldier: r_hat = r / (||r|| + eps)
        3. Perpendicular unit vector: r_perp = [-r_hat[1], r_hat[0]]
        4. Update weave bias (AR(1)): a <- rho * a + sigma_a * eta, eta ~ N(0,1)
        5. Heading noise: z ~ N(0, I_2)
        6. Unnormalized direction: u_raw = r_hat + a * r_perp + sigma_e * z
        7. Normalize: u = u_raw / (||u_raw|| + eps)
        8. Move: e_next = e + v_e * dt * u
        9. Apply reflecting boundary conditions
        """
        eps = self.config.eps
        
        # Vector from enemy to soldier
        r = self._soldier_pos - self._enemy_pos
        r_norm = np.linalg.norm(r)
        
        # Unit vector toward soldier
        if r_norm > eps:
            r_hat = r / r_norm
        else:
            # Enemy is on top of soldier, pick random direction
            angle = self._np_random.uniform(0, 2 * np.pi)
            r_hat = np.array([np.cos(angle), np.sin(angle)])
        
        # Perpendicular unit vector (90° CCW rotation)
        r_perp = np.array([-r_hat[1], r_hat[0]])
        
        # Update weave bias with AR(1) process: a <- rho * a + sigma_a * eta
        eta = self._np_random.normal(0.0, 1.0)
        self._weave_bias = self.config.rho * self._weave_bias + self.config.sigma_a * eta
        
        # Heading noise
        z = self._np_random.normal(0.0, 1.0, size=(2,))
        
        # Unnormalized direction with weave amplitude multiplier
        weave_component = self._weave_bias * self.config.weave_amplitude * r_perp
        u_raw = r_hat + weave_component + self.config.sigma_e * z
        
        # Normalize direction
        u_norm = np.linalg.norm(u_raw)
        if u_norm > eps:
            u = u_raw / u_norm
        else:
            u = r_hat  # Fallback to direct pursuit
        
        # Move enemy
        displacement = self.config.v_e * self.config.dt * u
        new_pos = self._enemy_pos + displacement
        
        # Apply reflecting boundary conditions
        new_pos = self._reflect_boundary(new_pos)
        
        self._enemy_pos = new_pos.astype(np.float32)
    
    def _reflect_boundary(self, pos: np.ndarray) -> np.ndarray:
        """
        Apply reflecting boundary conditions at [-L, L]².
        
        If position exceeds boundary, reflect it back into domain.
        
        Args:
            pos: Position array of shape (2,).
        
        Returns:
            Reflected position within [-L, L]².
        """
        L = self.config.L
        
        for i in range(2):
            # Reflect until within bounds (handles multiple reflections)
            while pos[i] < -L or pos[i] > L:
                if pos[i] < -L:
                    pos[i] = -2 * L - pos[i]  # Reflect off lower boundary
                if pos[i] > L:
                    pos[i] = 2 * L - pos[i]  # Reflect off upper boundary
        
        return pos
    
    def _get_obs(self) -> np.ndarray:
        """
        Get normalized observation for RL.
        
        Returns:
            obs: Array of shape (8,) with:
                [soldier_x, soldier_y, defender_x, defender_y, 
                 enemy_x, enemy_y, defender_to_enemy_x, defender_to_enemy_y]
                All values normalized to [-1, 1] by dividing positions by L.
        """
        L = self.config.L
        
        # Normalize positions to [-1, 1]
        soldier_norm = self._soldier_pos / L
        defender_norm = self._defender_pos / L
        enemy_norm = self._enemy_pos / L
        
        # Relative vector from defender to enemy (normalized)
        defender_to_enemy = (self._enemy_pos - self._defender_pos) / L
        # Clip to [-1, 1] in case entities are far apart
        defender_to_enemy = np.clip(defender_to_enemy, -1.0, 1.0)
        
        return np.concatenate([
            soldier_norm, 
            defender_norm, 
            enemy_norm,
            defender_to_enemy
        ]).astype(np.float32)
    
    def _get_info(self, outcome: str = "ongoing", enemy_soldier_dist: float = 0.0, 
                  defender_enemy_dist: float = 0.0) -> dict:
        """Get additional info dict.
        
        Args:
            outcome: Episode outcome - "ongoing", "intercepted", "soldier_caught", 
                    "collision_loss", or "timeout".
            enemy_soldier_dist: Distance between enemy and soldier.
            defender_enemy_dist: Distance between defender and enemy.
        """
        return {
            "step_count": self._step_count,
            "soldier_pos": self._soldier_pos.copy(),
            "defender_pos": self._defender_pos.copy(),
            "enemy_pos": self._enemy_pos.copy(),
            "weave_bias": self._weave_bias,
            "enemy_soldier_dist": enemy_soldier_dist,
            "defender_enemy_dist": defender_enemy_dist,
            "outcome": outcome,
        }
    
    def render(self) -> np.ndarray | None:
        """Render the environment (placeholder for future implementation)."""
        if self.render_mode == "rgb_array":
            # Return a simple representation (to be expanded later)
            return self._render_frame()
        return None
    
    def _render_frame(self) -> np.ndarray:
        """Render a frame as an RGB array."""
        # Simple placeholder: 100x100 black image with entities
        size = 100
        frame = np.zeros((size, size, 3), dtype=np.uint8)
        L = self.config.L
        
        # Draw soldier as blue square
        if self._soldier_pos is not None:
            x = int((self._soldier_pos[0] + L) / (2 * L) * (size - 1))
            y = int((self._soldier_pos[1] + L) / (2 * L) * (size - 1))
            x = np.clip(x, 0, size - 1)
            y = np.clip(y, 0, size - 1)
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    px, py = x + dx, y + dy
                    if 0 <= px < size and 0 <= py < size:
                        frame[py, px] = [0, 0, 255]  # Blue
        
        # Draw defender as green square
        if self._defender_pos is not None:
            x = int((self._defender_pos[0] + L) / (2 * L) * (size - 1))
            y = int((self._defender_pos[1] + L) / (2 * L) * (size - 1))
            x = np.clip(x, 0, size - 1)
            y = np.clip(y, 0, size - 1)
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    px, py = x + dx, y + dy
                    if 0 <= px < size and 0 <= py < size:
                        frame[py, px] = [0, 255, 0]  # Green
        
        # Draw enemy as red square
        if self._enemy_pos is not None:
            x = int((self._enemy_pos[0] + L) / (2 * L) * (size - 1))
            y = int((self._enemy_pos[1] + L) / (2 * L) * (size - 1))
            x = np.clip(x, 0, size - 1)
            y = np.clip(y, 0, size - 1)
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    px, py = x + dx, y + dy
                    if 0 <= px < size and 0 <= py < size:
                        frame[py, px] = [255, 0, 0]  # Red
        
        return frame
    
    def close(self) -> None:
        """Clean up resources."""
        pass
