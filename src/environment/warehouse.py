import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Tuple, Dict, Any

class Obstacle:
    """Base class for obstacles."""
    def get_signed_distance(self, p: np.ndarray) -> float:
        raise NotImplementedError

    def is_inside(self, p: np.ndarray) -> bool:
        return self.get_signed_distance(p) <= 0

    def plot(self, ax):
        raise NotImplementedError

class CircleObstacle(Obstacle):
    """Circular obstacle."""
    def __init__(self, center: Tuple[float, float], radius: float):
        self.center = np.array(center)
        self.radius = radius

    def get_signed_distance(self, p: np.ndarray) -> float:
        if p.ndim == 1:
            p = p.reshape(1, 2)
        dist = np.linalg.norm(p - self.center, axis=1) - self.radius
        return dist.squeeze()

    def plot(self, ax):
        circle = plt.Circle(self.center, self.radius, color='gray', zorder=2)
        ax.add_patch(circle)

class BoxObstacle(Obstacle):
    """Axis-aligned rectangular obstacle."""
    def __init__(self, center: Tuple[float, float], size: Tuple[float, float]):
        self.center = np.array(center)
        self.half_size = np.array(size) / 2.0
        self.min_corner = self.center - self.half_size
        self.max_corner = self.center + self.half_size

    def get_signed_distance(self, p: np.ndarray) -> float:
        if p.ndim == 1:
            p = p.reshape(1, 2)
        
        # Correct SDF calculation
        d = np.abs(p - self.center) - self.half_size
        dist_outside = np.linalg.norm(np.maximum(d, 0), axis=1)
        dist_inside = np.minimum(0, np.max(d, axis=1))
        
        return (dist_outside + dist_inside).squeeze()


    def plot(self, ax):
        rect = plt.Rectangle(self.min_corner, 
                             self.half_size[0] * 2, 
                             self.half_size[1] * 2, 
                             color='gray', zorder=2)
        ax.add_patch(rect)


class WarehouseEnv:
    """
    Simulates the warehouse navigation task with a unicycle model.
    Internal State s = [x, y, theta, v]
    NN State s_nn = [x, y, cos(theta), sin(theta), v]
    Action a = [linear_accel, angular_velocity]
    """
    def __init__(self, config: Dict[str, Any]):
        # Store the full config, not just the 'env' sub-dictionary
        self.full_config = config
        self.env_config = config['env']
        
        self.workspace = np.array(self.env_config['workspace'])
        self.dt = self.env_config['dt']
        self.v_max = self.env_config['v_max']
        self.a_max = self.env_config['a_max']
        self.omega_max = self.env_config['omega_max']
        self.robot_radius = self.env_config['robot_radius']
        
        # Internal state: [x, y, theta, v]
        self.state: np.ndarray = None
        self.state_dim_internal = 4
        self.state_dim_nn = 5 # [x, y, cos(theta), sin(theta), v]
        self.action_dim = 2 # [a, omega]
        
        self.obstacles: List[Obstacle] = self._create_obstacles()

    def _create_obstacles(self) -> List[Obstacle]:
        """Defines the static obstacles in the warehouse."""
        obstacles = [
            BoxObstacle(center=(self.workspace[0]/2, -0.1), size=(self.workspace[0], 0.2)), # Bottom
            BoxObstacle(center=(self.workspace[0]/2, self.workspace[1]+0.1), size=(self.workspace[0], 0.2)), # Top
            BoxObstacle(center=(-0.1, self.workspace[1]/2), size=(0.2, self.workspace[1])), # Left
            BoxObstacle(center=(self.workspace[0]+0.1, self.workspace[1]/2), size=(0.2, self.workspace[1])), # Right
            
            BoxObstacle(center=(3.0, 3.0), size=(2.0, 1.0)),
            BoxObstacle(center=(3.0, 7.0), size=(2.0, 1.0)),
            BoxObstacle(center=(8.0, 5.0), size=(1.5, 3.0)),
            
            CircleObstacle(center=(6.0, 2.0), radius=0.5),
            CircleObstacle(center=(6.0, 8.0), radius=0.5),
        ]
        return obstacles

    def reset(self, start_pos: Tuple[float, float] = (1.0, 1.0)) -> np.ndarray:
        """Resets the robot to a start position, facing right."""
        # state = [x, y, theta, v]
        self.state = np.array([start_pos[0], start_pos[1], 0.0, 0.0])
        return self.get_nn_state(self.state)

    def get_nn_state(self, state: np.ndarray) -> np.ndarray:
        """Converts internal [x,y,theta,v] state to NN-friendly state."""
        if state.ndim == 1:
            x, y, theta, v = state
            return np.array([x, y, np.cos(theta), np.sin(theta), v])
        else:
            x, y, theta, v = state[:, 0], state[:, 1], state[:, 2], state[:, 3]
            return np.stack([x, y, np.cos(theta), np.sin(theta), v], axis=-1)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Applies a unicycle action and updates the environment state.
        Action a = [linear_accel, angular_velocity]
        """
        if self.state is None:
            raise ValueError("Environment must be reset before stepping.")
            
        # Clip action to limits
        lin_accel = np.clip(action[0], -self.a_max, self.a_max)
        ang_vel = np.clip(action[1], -self.omega_max, self.omega_max)
        
        x, y, theta, v = self.state
        
        # Euler integration for unicycle model
        v_next = v + lin_accel * self.dt
        v_next = np.clip(v_next, 0, self.v_max) # Unicycle model cannot move backwards
        
        theta_next = theta + ang_vel * self.dt
        
        # Standard unicycle kinematics
        x_next = x + v_next * np.cos(theta_next) * self.dt
        y_next = y + v_next * np.sin(theta_next) * self.dt
        
        self.state = np.array([x_next, y_next, theta_next, v_next])
        nn_state = self.get_nn_state(self.state)
        
        h_star, is_collision = self.get_ground_truth_safety(nn_state, return_bool=True)
        
        reward = 0.0 
        done = is_collision
        info = {'h_star': h_star, 'is_collision': is_collision}
        
        return nn_state, reward, done, info

    def get_ground_truth_safety(self, nn_state: np.ndarray, return_bool: bool = False) -> Any:
        """
        Calculates the ground-truth safety value h*(s).
        h*(s) = signed_distance_to_nearest_obstacle - robot_radius
        This function only needs position, which is the first 2 dims.
        """
        if nn_state.ndim == 1:
            pos = nn_state[:2]
        else:
            pos = nn_state[:, :2] # Handle batch of states

        min_dist = np.full(pos.shape[0] if pos.ndim > 1 else 1, np.inf)

        for obs in self.obstacles:
            dist = obs.get_signed_distance(pos)
            if pos.ndim > 1:
                min_dist = np.minimum(min_dist, dist)
            else:
                min_dist = min(min_dist, dist)
        
        h_star = min_dist - self.robot_radius
        
        if return_bool:
            is_collision = (h_star <= 0)
            return h_star.squeeze(), is_collision.squeeze()
        
        return h_star.squeeze()

    def get_ground_truth_feasibility(self, nn_state: np.ndarray, subgoal: np.ndarray) -> np.ndarray:
        """
        Calculates the ground-truth feasibility value V*(s, g).
        V*(s, g) = ||p - g||^2
        This function only needs position, which is the first 2 dims.
        """
        if nn_state.ndim == 1:
            pos = nn_state[:2]
        else:
            pos = nn_state[:, :2]
            
        if subgoal.ndim == 1:
            goal_pos = subgoal[:2]
        else:
            goal_pos = subgoal[:, :2]

        V_star = np.sum((pos - goal_pos) ** 2, axis=-1)
        return V_star.squeeze()

    def sample_random_state(self, near_boundary_of: float = None) -> np.ndarray:
        """
        Samples a random internal state [x, y, theta, 0].
        """
        while True:
            if near_boundary_of is not None:
                obs = np.random.choice(self.obstacles[4:])
                if isinstance(obs, CircleObstacle):
                    angle = np.random.rand() * 2 * np.pi
                    r = obs.radius + np.random.uniform(0, near_boundary_of)
                    pos = obs.center + np.array([np.cos(angle), np.sin(angle)]) * r
                elif isinstance(obs, BoxObstacle):
                    edge = np.random.choice(4)
                    if edge == 0: # Top
                        x = np.random.uniform(obs.min_corner[0], obs.max_corner[0])
                        y = obs.max_corner[1] + np.random.uniform(0, near_boundary_of)
                    elif edge == 1: # Bottom
                        x = np.random.uniform(obs.min_corner[0], obs.max_corner[0])
                        y = obs.min_corner[1] - np.random.uniform(0, near_boundary_of)
                    elif edge == 2: # Left
                        x = obs.min_corner[0] - np.random.uniform(0, near_boundary_of)
                        y = np.random.uniform(obs.min_corner[1], obs.max_corner[1])
                    else: # Right
                        x = obs.max_corner[0] + np.random.uniform(0, near_boundary_of)
                        y = np.random.uniform(obs.min_corner[1], obs.max_corner[1])
                    pos = np.array([x, y])
            else:
                pos = np.random.rand(2) * self.workspace

            theta = np.random.rand() * 2 * np.pi
            v = 0.0
            
            internal_state = np.array([pos[0], pos[1], theta, v])
            nn_state = self.get_nn_state(internal_state)
            h_star = self.get_ground_truth_safety(nn_state)
            
            if h_star > -self.robot_radius: # Allow sampling from just inside
                return internal_state

    def sample_random_action(self) -> np.ndarray:
        """Samples a random action [a, omega]."""
        a = np.random.uniform(-self.a_max, self.a_max)
        omega = np.random.uniform(-self.omega_max, self.omega_max)
        return np.array([a, omega])

    def render(self, ax, nn_state: np.ndarray = None, goal: np.ndarray = None, path: np.ndarray = None):
        """Renders the environment on a matplotlib axes."""
        if nn_state is None:
            if self.state is None: # Handle case where render is called before reset
                return
            nn_state = self.get_nn_state(self.state)
        
        # Convert nn_state [x,y,cos,sin,v] to renderable [x,y,theta]
        x, y, cos_th, sin_th, v = nn_state
        theta = np.arctan2(sin_th, cos_th)
        
        ax.clear()
        
        for obs in self.obstacles:
            obs.plot(ax)
            
        if path is not None:
            ax.plot(path[:, 0], path[:, 1], 'g-', alpha=0.5, zorder=1)
            
        if goal is not None:
            # --- THIS IS THE FIX ---
            # Access clf_epsilon from the 'train' section of the full config
            clf_epsilon = self.full_config['train']['clf_epsilon']
            goal_circle = plt.Circle(goal[:2], np.sqrt(clf_epsilon), 
                                     color='g', fill=True, zorder=3, alpha=0.3)
            # --- END FIX ---
            ax.add_patch(goal_circle)

        # Plot robot as an arrow
        arrow_len = self.robot_radius * 1.5
        ax.arrow(x, y, 
                 arrow_len * np.cos(theta), arrow_len * np.sin(theta),
                 head_width=self.robot_radius * 0.8, 
                 head_length=self.robot_radius * 0.7, 
                 fc='r', ec='r', zorder=4)
        
        # Plot robot collision boundary
        robot_circle = plt.Circle((x, y), self.robot_radius, color='r', fill=False, ls='--', zorder=3)
        ax.add_patch(robot_circle)

        ax.set_xlim(0, self.env_config['workspace'][0])
        ax.set_ylim(0, self.env_config['workspace'][1])
        ax.set_aspect('equal')
        ax.grid(True, zorder=0)