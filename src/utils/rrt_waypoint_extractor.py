"""
Utility to extract waypoints from RRT planner paths for use in FSM.

This module provides functions to:
1. Run RRT planner to get a path
2. Extract key waypoints from the path (simplify/sample)
3. Convert to format suitable for FSM config
"""

import numpy as np
from typing import List, Optional, Tuple
from src.baselines.rrt_planner import RRTPlanner
from src.environment.warehouse import WarehouseEnv


def extract_waypoints_from_path(
    path: np.ndarray,
    max_waypoints: int = 5,
    min_distance: float = 1.0
) -> List[np.ndarray]:
    """
    Extract key waypoints from an RRT path.
    
    Args:
        path: Array of shape (N, 2) with [x, y] positions
        max_waypoints: Maximum number of waypoints to extract
        min_distance: Minimum distance between waypoints
    
    Returns:
        List of waypoint positions [x, y]
    """
    if path is None or len(path) == 0:
        return []
    
    if len(path) <= max_waypoints:
        # Path is short enough, return all points
        return [path[i] for i in range(len(path))]
    
    waypoints = []
    
    # Always include start
    waypoints.append(path[0])
    
    # Use Douglas-Peucker-like simplification
    # Or simple distance-based sampling
    last_waypoint = path[0]
    
    for i in range(1, len(path) - 1):
        dist = np.linalg.norm(path[i] - last_waypoint)
        
        # Add waypoint if far enough from last one
        if dist >= min_distance:
            waypoints.append(path[i])
            last_waypoint = path[i]
            
            # Stop if we have enough waypoints
            if len(waypoints) >= max_waypoints - 1:  # -1 to save room for goal
                break
    
    # Always include goal
    if len(waypoints) == 0 or np.linalg.norm(path[-1] - waypoints[-1]) > 0.1:
        waypoints.append(path[-1])
    
    return waypoints


def get_rrt_waypoints(
    env: WarehouseEnv,
    start: Tuple[float, float],
    goal: Tuple[float, float],
    max_waypoints: int = 5,
    min_distance: float = 1.0,
    **rrt_kwargs
) -> Optional[List[np.ndarray]]:
    """
    Run RRT planner and extract waypoints.
    
    Args:
        env: Warehouse environment
        start: Start position [x, y]
        goal: Goal position [x, y]
        max_waypoints: Maximum number of waypoints to extract
        min_distance: Minimum distance between waypoints
        **rrt_kwargs: Additional arguments for RRTPlanner
    
    Returns:
        List of waypoint positions [x, y], or None if planning fails
    """
    planner = RRTPlanner(env, **rrt_kwargs)
    path = planner.plan(start, goal)
    
    if path is None:
        return None
    
    waypoints = extract_waypoints_from_path(path, max_waypoints, min_distance)
    return waypoints


def waypoints_to_config_format(waypoints: List[np.ndarray]) -> List[List[float]]:
    """
    Convert waypoints to format suitable for YAML config.
    
    Args:
        waypoints: List of numpy arrays [x, y]
    
    Returns:
        List of lists [[x, y], ...] for YAML
    """
    return [[float(wp[0]), float(wp[1])] for wp in waypoints]

