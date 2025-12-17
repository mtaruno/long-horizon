"""
Script to extract waypoints from RRT planner and update config.

Usage:
    python -m scripts.extract_rrt_waypoints --config config/warehouse_v1.yaml
"""

import yaml
import numpy as np
import argparse
from pathlib import Path

from src.environment.warehouse import WarehouseEnv
from src.utils.rrt_waypoint_extractor import get_rrt_waypoints, waypoints_to_config_format


def main():
    parser = argparse.ArgumentParser(description="Extract waypoints from RRT planner")
    parser.add_argument("--config", type=str, default="config/warehouse_v1.yaml",
                       help="Path to config file")
    parser.add_argument("--max-waypoints", type=int, default=5,
                       help="Maximum number of waypoints to extract")
    parser.add_argument("--min-distance", type=float, default=1.0,
                       help="Minimum distance between waypoints")
    parser.add_argument("--output", type=str, default=None,
                       help="Output config file (default: overwrite input)")
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize environment
    env = WarehouseEnv(config)
    
    # Get start and goal from config
    start = np.array(config['fsm']['start_state'])
    goal = np.array(config['fsm']['goal_state'])
    
    print(f"Planning path from {start} to {goal}...")
    
    # Run RRT planner
    waypoints = get_rrt_waypoints(
        env=env,
        start=start,
        goal=goal,
        max_waypoints=args.max_waypoints,
        min_distance=args.min_distance,
        step_size=0.5,
        max_iterations=5000,
        goal_radius=0.5
    )
    
    if waypoints is None:
        print("❌ RRT planning failed! Using existing waypoints from config.")
        return
    
    # Remove start and goal (FSM will add them automatically)
    waypoints = waypoints[1:-1]  # Remove first (start) and last (goal)
    
    print(f"\n✓ Found {len(waypoints)} waypoints:")
    for i, wp in enumerate(waypoints):
        print(f"  Waypoint {i+1}: [{wp[0]:.2f}, {wp[1]:.2f}]")
    
    # Update config
    config['fsm']['waypoints'] = waypoints_to_config_format(waypoints)
    
    # Save config
    output_path = args.output if args.output else args.config
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"\n✓ Updated config saved to {output_path}")


if __name__ == "__main__":
    main()


