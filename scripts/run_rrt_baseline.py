import argparse
import os
from pathlib import Path
import yaml
import numpy as np

from src.environment import WarehouseEnv
from src.baselines import RRTPlanner, PathFollowingController
from src.utils.visualization import EnvironmentVisualizer, create_evaluation_animation

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the geometric RRT baseline.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/warehouse_v1.yaml",
        help="Path to a warehouse config file.",
    )
    parser.add_argument(
        "--plot-path",
        type=str,
        default="visualizations/rrt_baseline_path.png",
        help="Where to save the static trajectory plot.",
    )
    parser.add_argument(
        "--gif-path",
        type=str,
        default="visualizations/rrt_baseline.gif",
        help="Where to save the rollout animation (set to '' to skip).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for the RRT sampler.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    with open(args.config, "r") as fp:
        config = yaml.safe_load(fp)

    env = WarehouseEnv(config)

    planner = RRTPlanner(
        env,
        step_size=0.7,
        goal_radius=0.4,
        goal_sample_rate=0.2,
    )
    controller = PathFollowingController(
        env,
        waypoint_tolerance=0.25,
        target_speed=1.2,
        kp_heading=3.0,
    )

    start_xy = np.array(config["fsm"]["start_state"][:2], dtype=np.float32)
    goal_xy = np.array(config["fsm"]["goal_state"][:2], dtype=np.float32)

    print("Planning with RRT...")
    path = planner.plan(start_xy, goal_xy)
    if path is None:
        raise RuntimeError("Failed to find a collision-free path with RRT.")

    print(f"Planned path with {len(path)} waypoints.")
    result = controller.rollout(path)
    print(f"Reached goal: {result.reached_goal}, Collided: {result.collided}")
    print(f"Executed {len(result.actions)} actions / {len(result.executed_states)} recorded states.")

    vis = EnvironmentVisualizer(env)
    fig, ax = vis.plot_environment(goal=config["fsm"]["goal_state"], show_labels=False)
    
    # Plot the RRT-planned path (waypoints)
    ax.plot(path[:, 0], path[:, 1], 'b--', linewidth=2, alpha=0.6, label='RRT Planned Path', zorder=3)
    ax.scatter(path[:, 0], path[:, 1], c='blue', s=30, marker='o', alpha=0.7, 
               edgecolors='darkblue', linewidths=1, label='RRT Waypoints', zorder=4)
    
    # Plot the executed trajectory
    vis.plot_trajectory(result.executed_states, ax=ax, color="orange", label="Executed Trajectory")
    ax.set_title("RRT Baseline: Planned Path vs Executed Trajectory")

    plot_path = Path(args.plot_path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path)
    print(f"Saved trajectory plot to {plot_path}")

    if args.gif_path:
        gif_path = Path(args.gif_path)
        gif_path.parent.mkdir(parents=True, exist_ok=True)
        create_evaluation_animation(env, result.executed_states, config["fsm"]["goal_state"], str(gif_path))
        print(f"Saved rollout GIF to {gif_path}")


if __name__ == "__main__":
    main()

