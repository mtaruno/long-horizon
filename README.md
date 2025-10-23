# Long-Horizon Safe & Feasible Planning

A PyTorch implementation of neural Control Barrier Functions (CBFs) and Control Lyapunov Functions (CLFs) for safe and feasible long-horizon planning in robotics.

## Overview

This framework learns safety and feasibility constraints from data to enable safe robot control over long horizons. It combines three key components:
- **Neural CBFs** - Learn safety constraints to avoid unsafe states
- **Neural CLFs** - Learn feasibility constraints to reach goal states
- **Ensemble Dynamics** - Learn system dynamics with uncertainty quantification

## Key Features
- **Real-time Safety**: CBF-CLF controller filters unsafe actions
- **Uncertainty-Aware**: Ensemble models provide epistemic uncertainty
- **Minimal Integration**: Drop-in safety layer for existing policies
- **Isaac Gym Ready**: Designed for parallel simulation environments

## Quick Start

```bash
# Setup
./setup.sh
source venv/bin/activate

# Run example
python example.py
```

## Usage
```python
from src import create_trainer

# Create trainer
trainer = create_trainer(state_dim=4, action_dim=2)

# Add training data
trainer.add_transition(
    state=state, action=action, next_state=next_state,
    is_safe=True, is_goal=False
)

# Get safe actions
safe_action = trainer.get_safe_action(state, proposed_action)
```

## Isaac Gym Integration

```python
# In your training loop
for step in range(max_steps):
    obs = env.get_observations()
    actions = policy(obs)
    
    # Filter through safety constraints
    safe_actions = trainer.get_safe_action(obs, actions)
    
    next_obs, rewards, dones, info = env.step(safe_actions)
    
    # Learn constraints from simulation
    trainer.add_transition(
        state=obs.cpu().numpy(),
        action=safe_actions.cpu().numpy(),
        next_state=next_obs.cpu().numpy(),
        is_safe=~info['collision'],
        is_goal=info['success']
    )
```



## Architecture

```
Environment → CBF-CLF Controller → Safe Actions
     ↓              ↑
Training Data → Constraint Learning
     ↓              ↑
Dynamics Model ← Uncertainty Estimation
```

## Components
- `cbf.py` - Control Barrier Functions for safety
- `clf.py` - Control Lyapunov Functions for feasibility  
- `models.py` - Ensemble dynamics learning
- `main_trainer.py` - Integrated training framework
- `example.py` - Usage demonstration
