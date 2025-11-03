# Quick Start Guide

## ✅ Your CBF/CLF Implementation is CORRECT

Your mathematical formulations in `cbf.py` and `clf.py` match the paper exactly. No changes needed there.

## 🆕 What Was Added

Three new modular components to complete the framework:

1. **`src/core/policy.py`** - Subgoal-conditioned policy π_θ(s, g)
2. **`src/planning/fsm_planner.py`** - FSM with Algorithm 1 (pruning)
3. **`src/training/integrated_trainer.py`** - Algorithm 2 (joint training)

## 🚀 Run the Example

```bash
python example_minimal.py
```

Expected output:
```
============================================================
Minimal FSM-CBF-CLF Framework Example
============================================================

✓ Environment created (goal: [10.  8.])
✓ FSM created with 3 states
✓ Neural networks initialized
✓ Optimizers created
✓ Replay buffer initialized
✓ Integrated trainer created

============================================================
Starting Training (Algorithm 2)
============================================================

Episode 1/5:
  Reward: -245.32
  Steps: 200
  Goal Reached: False
  Buffer Size: 200
...
```

## 📖 Understanding the Architecture

### High-Level Flow

```
FSM State Machine
      ↓ (provides subgoal g)
Policy π_θ(s, g)
      ↓ (proposes action a)
CBF-CLF Penalties
      ↓ (safe action)
Environment
      ↓ (next state s')
Learning Updates
```

### Training Loop (Algorithm 2)

Each episode:
1. **FSM** provides current subgoal based on state
2. **Policy** proposes action: `a = π_θ(state, subgoal)`
3. **Execute** action in environment
4. **Label** data (safe/unsafe, goal/non-goal)
5. **Update** dynamics, CBF, CLF, policy periodically
6. **Transition** FSM when predicates satisfied

### FSM Pruning (Algorithm 1)

After training:
```python
pruned_fsm = trainer.prune_fsm()
```

Removes:
- States where `h(s) < 0` (unsafe)
- Transitions where `V(s') > ε` (infeasible)

## 🔧 Customization

### Create Your Own FSM

```python
from src.planning.fsm_planner import FSMState, FSMTransition, FSMAutomaton

# Define states with subgoals
states = [
    FSMState(id="start", subgoal=np.array([0, 0]), is_goal=False),
    FSMState(id="waypoint", subgoal=np.array([5, 5]), is_goal=False),
    FSMState(id="goal", subgoal=np.array([10, 10]), is_goal=True)
]

# Define transitions with predicates
transitions = [
    FSMTransition("start", "waypoint", "reached_waypoint"),
    FSMTransition("waypoint", "goal", "reached_goal")
]

# Define predicate functions
def reached_waypoint(state):
    return np.linalg.norm(state[:2] - np.array([5, 5])) < 0.5

def reached_goal(state):
    return np.linalg.norm(state[:2] - np.array([10, 10])) < 0.5

predicates = {
    "reached_waypoint": reached_waypoint,
    "reached_goal": reached_goal
}

# Create FSM
fsm = FSMAutomaton(states, transitions, "start", predicates)
```

### Adjust Training Config

```python
config = {
    "lambda_cbf": 2.0,      # Increase for stricter safety
    "lambda_clf": 1.5,      # Increase for faster convergence
    "epsilon": 0.15,        # CLF goal threshold
    "batch_size": 128,      # Larger for more stable training
    "model_update_freq": 3, # More frequent dynamics updates
    "cbf_update_freq": 5,   # More frequent safety updates
    "clf_update_freq": 5    # More frequent feasibility updates
}
```

### Use Your Own Environment

Your environment needs:
```python
class YourEnv:
    def reset(self) -> np.ndarray:
        """Return initial state"""
        
    def step(self, action: np.ndarray) -> tuple:
        """
        Returns:
            next_state: np.ndarray
            reward: float
            done: bool
            info: dict with 'collision' and 'success' keys
        """
```

## 📊 Key Components

| Component | File | Purpose |
|-----------|------|---------|
| CBF | `src/cbf.py` | Safety certificates h(s) ≥ 0 |
| CLF | `src/clf.py` | Feasibility certificates V(s) → 0 |
| Dynamics | `src/models.py` | Learn P̂(s,a) → s' |
| Policy | `src/core/policy.py` | Learn π_θ(s,g) → a |
| FSM | `src/planning/fsm_planner.py` | High-level task structure |
| Trainer | `src/training/integrated_trainer.py` | Algorithm 2 |

## 🎯 What's Different from Your Original Code

### Before
- CBF/CLF: ✅ Correct but standalone
- No policy network
- FSM: Warehouse-specific
- Training: Separate, not integrated
- No FSM pruning

### After
- CBF/CLF: ✅ Same (kept your correct implementation)
- Policy: 🆕 Subgoal-conditioned π_θ(s,g)
- FSM: 🆕 Generic with pruning (Algorithm 1)
- Training: 🆕 Integrated (Algorithm 2)
- Structure: 🆕 Modular (core/planning/training)

## 💡 Next Steps

1. **Test**: Run `example_minimal.py`
2. **Customize**: Create your own FSM for your task
3. **Integrate**: Use with your environment
4. **Extend**: Add LTL compiler, QP solver, etc.

## 📚 Files to Read

1. `IMPLEMENTATION_SUMMARY.md` - Detailed technical summary
2. `REFACTOR_PLAN.md` - Full refactoring roadmap
3. `example_minimal.py` - Working example
4. `src/core/policy.py` - Policy implementation
5. `src/planning/fsm_planner.py` - FSM implementation
6. `src/training/integrated_trainer.py` - Training loop

## ❓ Common Questions

**Q: Do I need to change my CBF/CLF code?**  
A: No! Your implementation is correct.

**Q: Can I use my existing dynamics model?**  
A: Yes, just pass it to `FSMCBFCLFTrainer`.

**Q: How do I add more FSM states?**  
A: Add more `FSMState` objects with their subgoals and predicates.

**Q: What if I don't have an FSM yet?**  
A: Use `create_simple_navigation_fsm()` as a starting point.

**Q: Can I use this with Isaac Gym?**  
A: Yes! Just wrap your Isaac Gym env to match the interface.
