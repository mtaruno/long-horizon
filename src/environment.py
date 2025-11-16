import gymnasium as gym
from gymnasium import spaces
import numpy as np

class WarehouseEnvironment(gym.Env):
    """
    Implements the 2D warehouse environment with unicycle dynamics.
    Based on standard unicycle robot model.

    State: [x, y, θ, v]  (position, heading, linear velocity)
    Action: [a, ω]       (linear acceleration, angular velocity)
    """
    def __init__(self, goal_pos=np.array([10.5, 8.5]), goal_radius=0.4):
        super().__init__()

        # Physics parameters for unicycle model
        self.dt = 0.1
        self.max_vel = 1.0          # Maximum linear velocity (m/s)
        self.max_accel = 5.0        # Maximum linear acceleration (m/s²)
        self.max_omega = 2.0        # Maximum angular velocity (rad/s)
        self.safety_margin = 0.1

        # Environment layout
        self.bounds = np.array([12.0, 10.0])
        self.obstacles = [
            {'center': np.array([2.0, 3.0]), 'radius': 0.8},
            {'center': np.array([2.0, 7.0]), 'radius': 0.8},
            {'center': np.array([5.0, 2.0]), 'radius': 0.6},
            {'center': np.array([5.0, 5.0]), 'radius': 0.6},
            {'center': np.array([5.0, 8.0]), 'radius': 0.6},
            {'center': np.array([7.0, 3.5]), 'radius': 0.5},
            {'center': np.array([7.0, 6.5]), 'radius': 0.5},
            {'center': np.array([9.0, 1.0]), 'radius': 0.7},
            {'center': np.array([9.0, 5.0]), 'radius': 0.7},
            {'center': np.array([9.0, 9.0]), 'radius': 0.7},
            {'center': np.array([11.0, 3.0]), 'radius': 0.5},
        ]

        # Primary goal for this environment
        self.target_goal = {'center': goal_pos, 'radius': goal_radius}

        # Gym spaces
        # Actions: [a, ω] - linear acceleration and angular velocity
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )

        # State space: [x, y, θ, v]
        low_obs = np.array([0.0, 0.0, -np.pi, 0.0])
        high_obs = np.array([self.bounds[0], self.bounds[1], np.pi, self.max_vel])
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)

        self.state = None

    def _is_safe(self, state):
        """Checks if the state is outside all obstacles + safety margin."""
        pos = state[:2]
        for obs in self.obstacles:
            dist = np.linalg.norm(pos - obs['center'])
            # [cite: 186]
            if dist < (obs['radius'] + self.safety_margin):
                return False
        return True

    def _is_goal(self, state):
        """Checks if the state is within the target goal region."""
        pos = state[:2]
        dist_to_goal = np.linalg.norm(pos - self.target_goal['center'])
        return dist_to_goal <= self.target_goal['radius']

    def reset(self, seed=None, options=None):
        if seed is not None:
            super().reset(seed=seed)

        # Use fixed start position for consistent training with waypoints
        # Start position: bottom-left area, safe from obstacles
        fixed_start_pos = np.array([0.5, 10.0])
        fixed_start_heading = 0.0  # Facing right (positive x direction)

        # Check if we want random start (can be controlled via options)
        use_random_start = options.get('random_start', False) if options else False

        if use_random_start:
            # Start at a random safe position with zero velocity and random heading
            while True:
                pos = self.observation_space.sample()[:2]
                theta = np.random.uniform(-np.pi, np.pi)
                state = np.array([pos[0], pos[1], theta, 0.0])
                if self._is_safe(state):
                    self.state = state
                    break
        else:
            # Use fixed start position: [x, y, θ, v]
            self.state = np.array([fixed_start_pos[0], fixed_start_pos[1],
                                   fixed_start_heading, 0.0])

        info = {'is_safe': True, 'is_goal': self._is_goal(self.state)}
        return self.state, info

    def step(self, action):
        if self.state is None:
            raise RuntimeError("Must call reset() before step()")

        x, y, theta, v = self.state

        # Clip and scale actions
        # action[0] = linear acceleration (scaled to max_accel)
        # action[1] = angular velocity (scaled to max_omega)
        a = np.clip(action[0], -1.0, 1.0) * self.max_accel
        omega = np.clip(action[1], -1.0, 1.0) * self.max_omega

        # Unicycle dynamics (Forward Euler integration)
        # v_{k+1} = v_k + a * dt
        # θ_{k+1} = θ_k + ω * dt
        # x_{k+1} = x_k + v_{k+1} * cos(θ_{k+1}) * dt
        # y_{k+1} = y_k + v_{k+1} * sin(θ_{k+1}) * dt

        new_v = v + a * self.dt
        new_v = np.clip(new_v, 0.0, self.max_vel)  # Velocity is non-negative for unicycle

        new_theta = theta + omega * self.dt
        # Normalize angle to [-π, π]
        new_theta = np.arctan2(np.sin(new_theta), np.cos(new_theta))

        new_x = x + new_v * np.cos(new_theta) * self.dt
        new_y = y + new_v * np.sin(new_theta) * self.dt

        # Clip to bounds
        new_x = np.clip(new_x, 0.0, self.bounds[0])
        new_y = np.clip(new_y, 0.0, self.bounds[1])

        self.state = np.array([new_x, new_y, new_theta, new_v])

        # --- Check Status & Calculate Reward ---
        is_safe = self._is_safe(self.state)
        is_goal = self._is_goal(self.state)

        terminated = False # Episode ends (collision or goal)
        truncated = False  # Episode cut short (time limit)

        if not is_safe:
            reward = -10.0 # Collision penalty
            terminated = True
        elif is_goal:
            reward = 50.0  # Goal bonus
            terminated = True
        else:
            dist_to_goal = np.linalg.norm(self.state[:2] - self.target_goal['center'])
            reward = -dist_to_goal # Negative distance

        reward -= 0.01 * np.linalg.norm(action) # Small action penalty

        info = {
            'is_safe': is_safe,
            'is_goal': is_goal
        }

        return self.state, reward, terminated, truncated, info