"""Minimal Gymnasium environment with a stochastically moving soldier."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from uav_defend.config.env_config import EnvConfig
from kalman_tracker import KalmanTracker


class SoldierEnv(gym.Env):
    """
    A 2D discrete-time environment for RL training with partial observability.
    
    The defender drone is controlled by the RL agent to intercept
    an enemy drone before it reaches the soldier.
    
    Domain: Ω = [-L, L]² (square centered at origin).
    
    Entities:
        - Soldier (s ∈ ℝ²): Gaussian random walk (uncontrolled)
        - Defender (d ∈ ℝ²): Controlled by RL agent via 2D heading action
        - Enemy (e ∈ ℝ²): Weaving pursuit toward soldier (uncontrolled)
    
    Partial Observability with Kalman Tracking:
        - Before detection: Enemy state is MASKED (zeros)
        - After detection: Kalman filter estimates enemy state from noisy measurements
        - Detection occurs when defender is within detection_radius of enemy
        - TRUE enemy state is NEVER revealed to the RL agent
        - Defender motion is controlled entirely by external policy (action-driven)
    
    Observation (shape=9, normalized to [-1, 1]):
        [soldier_x, soldier_y, defender_x, defender_y, detected_flag,
         e_hat_x, e_hat_y, v_hat_x, v_hat_y]
        - detected_flag: 0.0 (not detected) or 1.0 (detected)
        - e_hat: estimated enemy position from Kalman filter (normalized by L)
        - v_hat: estimated enemy velocity from Kalman filter (normalized by v_e)
        - All values are 0.0 before detection
    
    Action (shape=2):
        2D continuous action vector in [-1, 1]².
        - Normalized to unit vector if norm > eps
        - Defender displacement = v_d * dt * normalized_action
        - If action norm < eps, defender does not move
        - Action controls defender heading at ALL times (policy-driven)
    
    Reward (dense shaping for RL):
        +100 for intercepting enemy safely (WIN)
        -100 for soldier caught (LOSS)
        -100 for unsafe intercept (intercept too close to soldier)
        -100 for timeout
        +5.0 * (prev_dist - curr_dist) for closing distance to enemy
        -0.05 per step (encourages efficiency)
    
    Termination:
        - Intercepted safely: dist_de < intercept_radius AND dist_es > unsafe_intercept_radius (WIN)
        - Soldier caught: dist_es < threat_radius (LOSS)
        - Unsafe intercept: dist_de < intercept_radius AND dist_es <= unsafe_intercept_radius (LOSS)
        - Timeout: step_count >= max_steps
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(
        self,
        config: EnvConfig | None = None,
        render_mode: str | None = None,
        use_kalman_obs: bool = True,
    ):
        """
        Initialize the SoldierEnv.
        
        Args:
            config: Environment configuration. Uses defaults if None.
            render_mode: One of "human", "rgb_array", or None.
            use_kalman_obs: If True, observation uses Kalman estimates (e_hat, v_hat).
                           If False, uses legacy format with true enemy position.
        """
        super().__init__()
        
        self.config = config if config is not None else EnvConfig()
        self.render_mode = render_mode
        self.use_kalman_obs = use_kalman_obs
        
        # Observation space: normalized positions + detection flag + enemy info
        # If use_kalman_obs=True (Kalman mode):
        #   [soldier(2), defender(2), detected_flag(1), e_hat(2), v_hat(2)]
        #   e_hat: estimated enemy position, v_hat: estimated velocity
        # If use_kalman_obs=False (Legacy mode):
        #   [soldier(2), defender(2), detected_flag(1), enemy_pos(2), rel_to_enemy(2)]
        #   enemy_pos: TRUE enemy position, rel_to_enemy: defender→enemy vector
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(9,),
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
        self._enemy_detected: bool = False  # Tracking state: enemy detected?
        self._np_random: np.random.Generator | None = None
        
        # Kalman filter for enemy tracking (initialized on first detection)
        self._kf: KalmanTracker | None = None
        self._e_hat: np.ndarray | None = None  # Estimated enemy position
        self._v_hat: np.ndarray | None = None  # Estimated enemy velocity
        self._prev_tracking_error: float | None = None  # For tracking error improvement reward
        
        # Measurement noise standard deviation for enemy position sensing
        self._measurement_noise_std: float = 0.5  # Lower = better Kalman tracking
    
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
        
        # Initialize detection state: enemy not detected at start
        self._enemy_detected = False
        
        # Reset Kalman filter state
        self._kf = None
        self._e_hat = None
        self._v_hat = None
        self._prev_tracking_error = None  # For tracking error improvement reward
        
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
            action: 2D continuous action vector in [-1, 1]².
                   Normalized to unit vector if norm > eps.
                   Controls defender heading at all times.
        
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
        
        # Check termination conditions using dist_de and dist_es
        dist_de = defender_enemy_dist  # defender to enemy
        dist_es = enemy_soldier_dist   # enemy to soldier
        
        # Check if defender is close enough to intercept
        can_intercept = dist_de <= self.config.intercept_radius
        
        # Check if intercept would be unsafe (too close to soldier)
        unsafe_zone = dist_es <= self.config.unsafe_intercept_radius
        
        # WIN: Safe intercept (defender catches enemy, far enough from soldier)
        intercepted = can_intercept and not unsafe_zone
        
        # LOSS: Enemy reaches soldier (threat zone)
        soldier_caught = dist_es <= self.config.threat_radius
        
        # LOSS: Unsafe intercept (caught enemy but too close to soldier - collateral risk)
        unsafe_intercept = can_intercept and unsafe_zone and not soldier_caught
        
        # Episode terminates on any terminal condition or timeout
        terminated = intercepted or soldier_caught or unsafe_intercept or (self._step_count >= self.config.max_steps)
        truncated = False
        
        # Determine outcome and reward
        if soldier_caught:
            # Check soldier_caught FIRST (highest priority failure)
            outcome = "soldier_caught"
            reward = self.config.reward_soldier_caught  # LOSS
        elif unsafe_intercept:
            # Unsafe intercept: caught enemy but too close to soldier
            outcome = "unsafe_intercept"
            reward = self.config.reward_unsafe_intercept  # LOSS
        elif intercepted:
            # Safe intercept (WIN)
            outcome = "intercepted"
            reward = self.config.reward_intercept  # WIN
        elif self._step_count >= self.config.max_steps:
            outcome = "timeout"
            reward = self.config.reward_timeout  # LOSS (failed to intercept)
        else:
            outcome = "ongoing"
            reward = 0.0
            
            # 1. Progress reward: positive for closing distance to enemy (using TRUE position)
            progress = self._prev_defender_enemy_dist - dist_de
            reward += self.config.reward_progress_scale * progress
            
            # 2. Small time penalty (encourages efficiency)
            reward += self.config.reward_time_penalty
            
            # 3. Tracking error improvement reward (only after detection)
            if self._enemy_detected and self._e_hat is not None:
                tracking_error = float(np.linalg.norm(self._enemy_pos - self._e_hat))
                if hasattr(self, '_prev_tracking_error') and self._prev_tracking_error is not None:
                    tracking_improvement = self._prev_tracking_error - tracking_error
                    reward += self.config.reward_tracking_scale * tracking_improvement
                self._prev_tracking_error = tracking_error
            
            # 4. Proximity warning: penalty when enemy gets close to soldier
            # Scaled by how close enemy is (inverse distance)
            proximity_threshold = self.config.unsafe_intercept_radius * 3.0  # ~10.5 units
            if dist_es < proximity_threshold:
                # Stronger penalty as enemy gets closer
                proximity_factor = 1.0 - (dist_es / proximity_threshold)
                reward += self.config.reward_proximity_warning * proximity_factor
            
            # Clip reward for numerical stability
            reward = np.clip(reward, -50.0, 50.0)
        
        # Update previous distance for next step
        self._prev_defender_enemy_dist = defender_enemy_dist
        
        return self._get_obs(), reward, terminated, truncated, self._get_info(
            outcome, enemy_soldier_dist, defender_enemy_dist
        )
    
    def _move_defender(self, action: np.ndarray) -> None:
        """
        Move the defender drone based on externally provided action.
        
        Args:
            action: 2D continuous action vector in [-1, 1]².
                   Normalized to unit vector if norm > eps.
                   Defender displacement = v_d * dt * normalized_action.
                   If action norm < eps, defender does not move.
        
        The defender is controlled entirely by the external policy.
        Detection and Kalman tracking are handled separately in _update_detection().
        """
        eps = self.config.eps
        
        # Update detection and Kalman tracking (independent of defender motion)
        self._update_detection()
        
        # Action-driven defender motion
        action = np.asarray(action, dtype=np.float32)
        action_norm = np.linalg.norm(action)
        
        # If action is too small, defender doesn't move
        if action_norm < eps:
            return
        
        # Normalize action to unit vector
        direction = action / action_norm
        
        # Move defender: displacement = v_d * dt * normalized_action
        displacement = self.config.v_d * self.config.dt * direction
        new_pos = self._defender_pos + displacement
        
        # Apply reflecting boundary conditions
        new_pos = self._reflect_boundary(new_pos)
        self._defender_pos = new_pos.astype(np.float32)
    
    def _update_detection(self) -> None:
        """
        Update enemy detection state and Kalman filter tracking.
        
        Detection occurs when defender is within detection_radius of enemy.
        Once detected, tracking persists and Kalman filter is updated each step.
        This is decoupled from defender motion to support external policies.
        """
        # Check for detection (only if not already detected)
        if not self._enemy_detected:
            defender_enemy_dist = np.linalg.norm(self._enemy_pos - self._defender_pos)
            if defender_enemy_dist <= self.config.detection_radius:
                self._enemy_detected = True  # Start tracking!
                
                # Initialize Kalman filter with first noisy measurement
                self._kf = KalmanTracker(dt=self.config.dt)
                noisy_measurement = self._enemy_pos + self._np_random.normal(
                    0.0, self._measurement_noise_std, size=(2,)
                )
                self._kf.initialize(noisy_measurement.astype(np.float64))
                self._e_hat = self._kf.get_position()
                self._v_hat = self._kf.get_velocity()
        else:
            # Already detected: predict and update Kalman filter
            self._kf.predict()
            noisy_measurement = self._enemy_pos + self._np_random.normal(
                0.0, self._measurement_noise_std, size=(2,)
            )
            self._kf.update(noisy_measurement.astype(np.float64))
            self._e_hat = self._kf.get_position()
            self._v_hat = self._kf.get_velocity()
    
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
        Get normalized observation for RL with partial observability.
        
        Returns:
            obs: Array of shape (9,).
            
            If use_kalman_obs=True (Kalman mode):
                [soldier_x, soldier_y, defender_x, defender_y, detected_flag,
                 e_hat_x, e_hat_y, v_hat_x, v_hat_y]
                - e_hat: Kalman estimated enemy position (normalized by L)
                - v_hat: Kalman estimated enemy velocity (normalized by v_e)
                
            If use_kalman_obs=False (Legacy mode):
                [soldier_x, soldier_y, defender_x, defender_y, detected_flag,
                 enemy_x, enemy_y, rel_x, rel_y]
                - enemy: TRUE enemy position (normalized by L)
                - rel: relative position from defender to enemy (normalized by L)
        """
        L = self.config.L
        v_e = self.config.v_e
        
        # Normalize positions to [-1, 1]
        soldier_norm = self._soldier_pos / L
        defender_norm = self._defender_pos / L
        
        # Detection flag
        detected_flag = np.array([1.0 if self._enemy_detected else 0.0])
        
        if self.use_kalman_obs:
            # KALMAN MODE: Use Kalman filter estimates
            if self._enemy_detected and self._e_hat is not None:
                e_hat_norm = np.clip(self._e_hat / L, -1.0, 1.0)
                v_hat_norm = np.clip(self._v_hat / v_e, -1.0, 1.0)
            else:
                e_hat_norm = np.array([0.0, 0.0])
                v_hat_norm = np.array([0.0, 0.0])
            
            return np.concatenate([
                soldier_norm, 
                defender_norm,
                detected_flag,
                e_hat_norm,
                v_hat_norm
            ]).astype(np.float32)
        else:
            # LEGACY MODE: Use true enemy position
            if self._enemy_detected:
                enemy_norm = self._enemy_pos / L
                defender_to_enemy = (self._enemy_pos - self._defender_pos) / L
                defender_to_enemy = np.clip(defender_to_enemy, -1.0, 1.0)
            else:
                enemy_norm = np.array([0.0, 0.0])
                defender_to_enemy = np.array([0.0, 0.0])
            
            return np.concatenate([
                soldier_norm, 
                defender_norm,
                detected_flag,
                enemy_norm,
                defender_to_enemy
            ]).astype(np.float32)
    
    def _get_info(self, outcome: str = "ongoing", enemy_soldier_dist: float = 0.0, 
                  defender_enemy_dist: float = 0.0) -> dict:
        """Get additional info dict.
        
        Args:
            outcome: Episode outcome - "ongoing", "intercepted", "soldier_caught", 
                    "unsafe_intercept", or "timeout".
            enemy_soldier_dist: Distance between enemy and soldier.
            defender_enemy_dist: Distance between defender and enemy.
        """
        # Check if current state is in unsafe intercept zone
        unsafe_zone = enemy_soldier_dist <= self.config.unsafe_intercept_radius
        
        # Compute tracking error if Kalman filter is active
        if self._e_hat is not None:
            tracking_error = float(np.linalg.norm(self._enemy_pos - self._e_hat))
        else:
            tracking_error = None
        
        return {
            "step_count": self._step_count,
            "soldier_pos": self._soldier_pos.copy(),
            "defender_pos": self._defender_pos.copy(),
            "enemy_pos": self._enemy_pos.copy(),
            "weave_bias": self._weave_bias,
            "enemy_soldier_dist": enemy_soldier_dist,
            "defender_enemy_dist": defender_enemy_dist,
            "enemy_detected": self._enemy_detected,  # Use stored tracking state
            "unsafe_intercept": unsafe_zone,  # True if enemy is in unsafe zone around soldier
            "outcome": outcome,
            # Kalman filter tracking info
            "detected": self._enemy_detected,
            "e_hat": self._e_hat.copy() if self._e_hat is not None else None,
            "v_hat": self._v_hat.copy() if self._v_hat is not None else None,
            "tracking_error": tracking_error,
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
