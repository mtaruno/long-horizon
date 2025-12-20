# Long-Horizon Safe & Feasible Planning





<!-- TODO: Show the TurtleBot Demo -->

## TLDR:
Problem: Long-horizon robotic navigation in cluttered environments is challenging due to the need to simultaneously ensure safety, feasibility, and temporal correctness over extended horizons. Classical planners (e.g., RRT) reason geometrically but ignore system dynamics and uncertainty, while reinforcement learning methods struggle with sparse rewards, safety violations, and poor generalization over long-horizon tasks.

Our Approach: We propose a modular long-horizon planning and control framework that combines high-level temporal planning with low-level safe and feasible policy learning. A Linear Temporal Logic (LTL) specification is compiled into a Finite State Machine (FSM) that decomposes the task into subgoals. A goal-conditioned neural policy is trained using both model-based and model-free signals, while learned Control Barrier Functions (CBFs) and Control Lyapunov Functions (CLFs) provide safety and progress certificates. During planning, we prune FSM transitions using these learned certificates, retaining only transitions that are predicted to be safe and feasible under the current policy and learned dynamics. We show that this works in a TurtleBot.

---

## Overview
Project structure:
```
long-horizon-planning/
├── config/                # Hyperparameters and environment settings
├── data/                  # Generated datasets
├── models/                # Saved model checkpoints (.pth)
├── notebooks/             # End-to-end and example notebooks
├── scripts/               # Executable entry points (training, evaluation)
└── src/                   # Core source code
    ├── core/             # Neural network architectures
    ├── environment/      # Physics, dynamics, and simulation
    ├── planning/         # High-level FSM planning logic
    ├── baselines/        # Classical controllers (RRT, etc.)
    └── utils/            # Buffers, logging, seeding helpers
```


The notebooks are where we show end-to-end examples for these modular components.


## Key Modules:

### Environment (src/environment/)
- warehouse.py: Defines the warehouse simulation environment.

Main Components
- Dynamics Model: Unicycle model
State: x, y, θ, v
- State Representation: 5D vector → `[x, y, cos(θ), sin(θ), v]`
- Action Space: 2D vector → `[linear_acceleration, angular_velocity]`
- Ground Truth Functions:
- `get_ground_truth_safety()` — Signed Distance Fields (SDF)
- `get_ground_truth_feasibility()` — Euclidean distance-based feasibility

### Core Networks (src/core/)

`critics.py`
Implements:
- CBFNetwork (safety critic)
- CLFNetwork (feasibility critic)

`policy.py`
Implements the SubgoalConditionedPolicy, featuring:
- Dual-constraint loss:
- Model-based gradients from dynamics ensemble
- Model-free gradients from real experience

`models.py`
Implements the EnsembleDynamicsModel:
- Ensemble of 5 neural networks for next-state prediction
- Used for uncertainty estimation & model-based training signals

### Planning (src/planning/)
`fsm_planner.py`
Implements the Hierarchical FSM planner

FSM Structure
`START → WAYPOINT_1 → GOAL`

Key Function — `prune_fsm_with_certificates()`
- Samples states and checks CBF/CLF certificates
- Transition kept only if ≥ 75% of samples pass safety + feasibility tests

## Execution Scripts

Execution Scripts (scripts/)
1.	generate_data.py: Generates pretraining dataset → `data/pretrain_data.pkl`.
2.	pretrain.py: Pretrains the CBF/CLF networks using offline samples.
3.	hpo_trainer.py: (Currently failing) Optuna-based hyperparameter search.
4.	evaluate.py: Loads champion model from models/best/ and runs visual demos.
5.	run_rrt_baseline.py: Deterministic RRT + path-following baseline for comparison / debugging.
    
Run this as: `python -m scripts.hpo_trainer`

To benchmark the geometric baseline, run:
```
python -m scripts.run_rrt_baseline --config config/warehouse_v1.yaml
```
This will plan a collision-free polyline with RRT, roll the WarehouseEnv forward
with a low-level path follower, and dump both a static figure and GIF in
`visualizations/`.

### Configuration (config/)
`warehouse_v1.yaml` — Single source of truth for entire pipeline.
Includes:
- Physics limits: v_max, omega_max
- Loss weights: lambda_cbf, lambda_clf
- Network sizes and training hyperparameters
- Reward/penalty shaping parameters
