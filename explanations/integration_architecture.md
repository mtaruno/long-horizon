# Hierarchical Control Architecture: FSM/LLM + CBF-CLF Integration

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    HIGH-LEVEL PLANNERS                          │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐              ┌─────────────────────────────┐│
│  │   FSM PLANNER   │              │      LLM PLANNER            ││
│  │                 │              │                             ││
│  │ • IDLE          │              │ "go to corner"              ││
│  │ • NAVIGATE      │              │      ↓                      ││
│  │ • AVOID         │              │ NLP Processing              ││
│  │ • PICKUP        │              │      ↓                      ││
│  │ • EMERGENCY     │              │ Semantic Understanding      ││
│  └─────────────────┘              └─────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 HIGH-LEVEL COMMANDS                             │
│  HighLevelCommand(                                              │
│    target_position=[3.0, 2.0],                                 │
│    target_velocity=[0.5, 0.3],                                 │
│    priority="efficiency"                                        │
│  )                                                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              LOW-LEVEL ACTION GENERATION                        │
│                                                                 │
│  PD Controller: action = Kp*(pos_error) + Kd*(vel_error)      │
│                                                                 │
│  Proposed Action: [ax, ay] = [1.42, 1.42] m/s²                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   SAFETY FILTER LAYER                          │
│                     (CBF-CLF Framework)                        │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │     CBF     │  │     CLF     │  │    DYNAMICS MODEL       │  │
│  │   Safety    │  │ Feasibility │  │   s' = f(s, a)         │  │
│  │ h(s) ≥ 0    │  │  V(s) → 0   │  │                        │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
│                              │                                  │
│                              ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │            CONSTRAINT CHECKER                           │   │
│  │  • h(s') - h(s) ≥ -α*h(s)  (Safety)                  │   │
│  │  • V(s') - V(s) ≤ -β*V(s)+δ (Feasibility)            │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                              ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │         ACTION MODIFICATION (if needed)                 │   │
│  │  • QP Solver (future)                                  │   │
│  │  • Gradient-based correction                           │   │
│  │  • Conservative fallback                               │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     SAFE ACTIONS                               │
│                                                                 │
│  Safe Action: [ax, ay] = [1.42, 1.42] m/s²                    │
│  (Minimal modification from proposed action)                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ROBOT EXECUTION                             │
│                                                                 │
│  Motor Commands → Physical Robot → State Update                │
└─────────────────────────────────────────────────────────────────┘
```

## Integration Flow with Actual Values

### **1. FSM Planner Example**
```python
# Input: Robot state + sensors
state = [x=1.0, y=1.0, vx=0.1, vy=0.1]
sensors = {"obstacle_detected": False}

# FSM Decision
fsm_state = "navigate_to_goal"
target_position = [3.0, 2.0]
target_velocity = [0.45, 0.22]

# PD Controller
pos_error = [3.0-1.0, 2.0-1.0] = [2.0, 1.0]
vel_error = [0.45-0.1, 0.22-0.1] = [0.35, 0.12]
proposed_action = 5.0*[2.0, 1.0] + 2.0*[0.35, 0.12] = [10.7, 5.24]

# Safety Filter
safe_action = cbf_clf_filter(state, proposed_action) = [1.80, 0.88]
# (Clipped to max acceleration limits)
```

### **2. LLM Planner Example**
```python
# Input: Natural language + robot state
command = "go to the corner"
state = [x=1.0, y=1.0, vx=0.1, vy=0.1]

# LLM Processing
parsed_intent = "navigate_to_position"
target_position = [5.0, 5.0]  # "corner" → coordinates
target_velocity = [0.8, 0.8]

# PD Controller  
proposed_action = [1.41, 1.41]  # Toward corner

# Safety Filter
safe_action = cbf_clf_filter(state, proposed_action) = [1.41, 1.41]
# (No modification needed - action is safe)
```

## Key Integration Benefits

### **1. Separation of Concerns**
- **High-level**: Task planning, goal selection, behavior logic
- **Low-level**: Safety constraints, feasibility, real-time control
- **Clean interface**: HighLevelCommand dataclass

### **2. Safety Guarantees**
- High-level planners can be **aggressive** or **suboptimal**
- Safety layer **always** ensures constraints satisfied
- Mathematical guarantees regardless of planner quality

### **3. Modularity**
- Swap FSM ↔ LLM ↔ other planners easily
- CBF-CLF framework remains unchanged
- Add new behaviors without safety re-engineering

### **4. Real-time Performance**
- High-level: 1-10 Hz (slower, complex reasoning)
- Safety filter: 100+ Hz (fast neural network inference)
- Decoupled update rates

## Practical Implementation Patterns

### **Pattern 1: Safety Override**
```python
def execute_with_safety(high_level_command, current_state):
    # Convert high-level to low-level
    proposed_action = pd_controller(high_level_command, current_state)
    
    # Safety filter
    safe_action = cbf_clf_filter(current_state, proposed_action)
    
    # Log if modified
    if not np.allclose(proposed_action, safe_action):
        log_safety_intervention(proposed_action, safe_action)
    
    return safe_action
```

### **Pattern 2: Adaptive Planning**
```python
def adaptive_planner(state, safety_metrics):
    if safety_metrics.safety_rate < 0.8:
        # Switch to conservative planner
        return conservative_fsm.plan(state)
    else:
        # Use aggressive LLM planner
        return llm_planner.plan(state)
```

### **Pattern 3: Hierarchical Feedback**
```python
def hierarchical_control_loop():
    while True:
        # High-level planning (1 Hz)
        if time % 1.0 == 0:
            high_level_cmd = planner.update(state, sensors)
        
        # Low-level safety control (100 Hz)
        proposed_action = pd_controller(high_level_cmd, state)
        safe_action = safety_filter(state, proposed_action)
        
        # Execute and update
        state = robot.execute(safe_action)
        
        # Learn from experience
        safety_trainer.add_transition(transition)
```

This architecture enables **safe autonomy at scale** by combining the flexibility of high-level reasoning (FSM/LLM) with the mathematical rigor of CBF-CLF safety constraints.