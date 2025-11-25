"""
Simple RRT-based baseline planner and path-following controller.

The planner works directly in the warehouse (x, y) workspace and relies on
the environment's ground-truth safety function for collision checks, so the
resulting polylines respect all obstacles with approximately the same
clearance as the learning-based agents.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple
import math
import random
import numpy as np
from src.environment.warehouse import WarehouseEnv


@dataclass
class Node:
    position: np.ndarray
    parent: Optional[int]


@dataclass
class BaselineRolloutResult:
    """Container for path, executed trajectory, and rollout info."""

    path: np.ndarray
    executed_states: List[np.ndarray]
    actions: List[np.ndarray]
    reached_goal: bool
    collided: bool


class RRTPlanner:
    """
    Rapidly-Exploring Random Tree (RRT) planner that respects the warehouse
    obstacles via signed-distance queries.
    """

    def __init__(
        self,
        env: WarehouseEnv,
        step_size: float = 0.5,
        max_iterations: int = 5000,
        goal_radius: float = 0.5,
        goal_sample_rate: float = 0.1,
        collision_check_resolution: float = 0.1,
        clearance_margin: float = 0.05,
    ):
        self.env = env
        self.step_size = step_size
        self.max_iterations = max_iterations
        self.goal_radius = goal_radius
        self.goal_sample_rate = goal_sample_rate
        self.collision_check_resolution = collision_check_resolution
        self.clearance_margin = clearance_margin
        self.workspace = env.workspace

    def plan(self, start: Sequence[float], goal: Sequence[float]) -> Optional[np.ndarray]:
        start = np.array(start, dtype=np.float32)
        goal = np.array(goal, dtype=np.float32)

        nodes: List[Node] = [Node(position=start, parent=None)]

        for iteration in range(self.max_iterations):
            sample = self._sample_free(goal)
            nearest_idx = self._nearest_node(nodes, sample)
            new_position = self._steer(nodes[nearest_idx].position, sample)

            if not self._segment_is_collision_free(nodes[nearest_idx].position, new_position):
                continue

            nodes.append(Node(position=new_position, parent=nearest_idx))

            if np.linalg.norm(new_position - goal) <= self.goal_radius:
                raw_path = self._reconstruct_path(nodes, len(nodes) - 1, goal)
                return self._densify_path(raw_path)

        return None

    def _sample_free(self, goal: np.ndarray) -> np.ndarray:
        if random.random() < self.goal_sample_rate:
            return goal

        return np.array(
            [
                random.uniform(0.0, self.workspace[0]),
                random.uniform(0.0, self.workspace[1]),
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _nearest_node(nodes: Sequence[Node], sample: np.ndarray) -> int:
        dists = [np.linalg.norm(node.position - sample) for node in nodes]
        return int(np.argmin(dists))

    def _steer(self, start: np.ndarray, target: np.ndarray) -> np.ndarray:
        direction = target - start
        dist = np.linalg.norm(direction)
        if dist <= self.step_size:
            return target
        return start + (direction / dist) * self.step_size

    def _segment_is_collision_free(self, p1: np.ndarray, p2: np.ndarray) -> bool:
        distance = np.linalg.norm(p2 - p1)
        steps = max(2, int(math.ceil(distance / self.collision_check_resolution)))

        for alpha in np.linspace(0.0, 1.0, steps):
            point = p1 + alpha * (p2 - p1)
            nn_state = np.array([point[0], point[1], 1.0, 0.0, 0.0], dtype=np.float32)

            h_star = self.env.get_ground_truth_safety(nn_state)
            if h_star <= 0.0:
                return False
            # if h_star <= self.clearance_margin:
            #     return False
        return True

    def _reconstruct_path(self, nodes: Sequence[Node], node_idx: int, goal: np.ndarray) -> np.ndarray:
        path = [goal]
        current_idx = node_idx

        while current_idx is not None:
            node = nodes[current_idx]
            path.append(node.position)
            current_idx = node.parent

        path.reverse()
        return np.vstack(path)

    def _densify_path(self, path: np.ndarray) -> np.ndarray:
        """Insert intermediate points so adjacent waypoints are close enough for the controller."""
        if len(path) <= 2:
            return path

        densified: List[np.ndarray] = [path[0]]
        max_segment = self.step_size / 2.0

        for idx in range(1, len(path)):
            start = path[idx - 1]
            end = path[idx]
            segment = end - start
            distance = np.linalg.norm(segment)

            if distance <= max_segment:
                densified.append(end)
                continue

            steps = int(math.ceil(distance / max_segment))
            for step in range(1, steps + 1):
                point = start + (segment * (step / steps))
                densified.append(point)

        return np.vstack(densified)


class PathFollowingController:
    """
    Deterministic controller that tracks a geometric path by producing
    accelerations and angular velocities compatible with WarehouseEnv.
    """

    def __init__(
        self,
        env: WarehouseEnv,
        waypoint_tolerance: float = 0.1,
        target_speed: float = 0.4,
        kp_heading: float = 4.0,
        max_steps: int = 10000,
    ):
        self.env = env
        self.waypoint_tolerance = waypoint_tolerance
        self.target_speed = target_speed
        self.kp_heading = kp_heading
        self.max_steps = max_steps

    def rollout(self, path: np.ndarray) -> BaselineRolloutResult:
        self.env.reset(start_pos=path[0])
        executed_states: List[np.ndarray] = []
        actions: List[np.ndarray] = []

        current_waypoint = 1
        reached_goal = False
        collided = False

        for _ in range(self.max_steps):
            nn_state = self.env.get_nn_state(self.env.state)
            executed_states.append(nn_state.copy())

            position = nn_state[:2]
            theta = math.atan2(nn_state[3], nn_state[2])
            speed = nn_state[-1]

            if current_waypoint >= len(path):
                reached_goal = True
                break

            target = path[current_waypoint]
            to_target = target - position
            distance = np.linalg.norm(to_target)

            if distance < self.waypoint_tolerance:
                current_waypoint += 1
                continue

            desired_heading = math.atan2(to_target[1], to_target[0])
            heading_error = self._wrap_angle(desired_heading - theta)
            angular_velocity = np.clip(self.kp_heading * heading_error, -self.env.omega_max, self.env.omega_max)

            desired_speed = min(self.target_speed, distance / max(self.env.dt, 1e-3))
            lin_accel = np.clip((desired_speed - speed) / self.env.dt, -self.env.a_max, self.env.a_max)

            action = np.array([lin_accel, angular_velocity], dtype=np.float32)
            nn_state_next, _, done, info = self.env.step(action)
            actions.append(action.copy())
            executed_states.append(nn_state_next.copy())

            collided = info["is_collision"]
            if collided:
                break

        if not collided:
            final_pos = executed_states[-1][:2]
            reached_goal = reached_goal or np.linalg.norm(final_pos - path[-1]) < self.waypoint_tolerance

        return BaselineRolloutResult(
            path=path,
            executed_states=executed_states,
            actions=actions,
            reached_goal=reached_goal,
            collided=collided,
        )

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        """Wrap angle to [-pi, pi]."""
        return (angle + math.pi) % (2 * math.pi) - math.pi

