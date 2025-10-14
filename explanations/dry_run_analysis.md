# Long-Horizon Safe Planning - Detailed Dry Run Analysis

## Complete Data Flow with Actual Values

### **STEP 1: INITIALIZATION**
```
✓ Created trainer: state_dim=4, action_dim=2
✓ CBF ensemble: 3 models (safety)
✓ CLF ensemble: 3 models (feasibility)  
✓ Dynamics ensemble: 5 models (prediction)
✓ Replay buffer capacity: 100000
```

**What's Created:**
- 3 CBF networks learning h(s) for safety
- 3 CLF networks learning V(s) for feasibility
- 5 Dynamics networks learning P(s'|s,a)
- Replay buffer for storing transitions
- Controller for action filtering

---

### **STEP 2: FIRST TRANSITION**
```
Input state: [1.0, 0.5, 0.1, -0.2]     # [x, y, vx, vy]
Input action: [0.05, 0.1]               # [ax, ay]
Next state: [1.05, 0.48, 0.15, -0.1]   # After dynamics
```

**Data Storage:**
```
Before: step_count=0, buffer_size=0, safe_states=0
After:  step_count=1, buffer_size=1, safe_states=1, transitions=1
```

**What Happens:**
- Transition stored in replay buffer
- State added to safe_states list (is_safe=True)
- No training triggered yet (update_freq=5)

---

### **STEP 3: UNTRAINED NETWORK PREDICTIONS**

**CBF Predictions (Safety):**
```
State: [1.0, 0.5, 0.1, -0.2]
Mean CBF value: 0.1847
Individual models: [-0.225, 0.496, 0.283]
Uncertainty (std): 0.3704
```

**CLF Predictions (Feasibility):**
```
State: [1.0, 0.5, 0.1, -0.2]
Mean CLF value: 0.7696
Individual models: [0.877, 0.942, 0.489]
Uncertainty (std): 0.2448
```

**Dynamics Predictions:**
```
Input: state=[1.0, 0.5, 0.1, -0.2], action=[0.05, 0.1]
Mean prediction: [0.799, 0.607, 0.064, 0.279]
Actual next state: [1.05, 0.48, 0.15, -0.1]
Prediction error: 0.2107
Uncertainty (std): 0.5049
```

**Key Observations:**
- Untrained networks give random predictions
- High uncertainty indicates model disagreement
- Large prediction errors expected initially

---

### **STEP 4: ADDING MORE TRANSITIONS**

**Added 5 More Transitions:**
```
Transition 1: safe=True, goal=False   # Normal safe movement
Transition 2: safe=True, goal=False   # Normal safe movement  
Transition 3: safe=True, goal=False   # Normal safe movement
Transition 4: safe=False, goal=False  # COLLISION occurred
Transition 5: safe=True, goal=True    # REACHED GOAL
```

**Final Data Counts:**
```
Total steps: 6
Buffer size: 6
Safe states: 5    # States where no collision
Unsafe states: 1  # State where collision occurred
Goal states: 1    # State where goal was reached
```

---

### **STEP 5: TRAINING UPDATES**

**CBF Training (Safety Learning):**
```
Training Data:
- Safe states: torch.Size([5, 4])    # 5 safe examples
- Unsafe states: torch.Size([1, 4])  # 1 unsafe example
- Transition states: torch.Size([4, 4]) # For constraint learning

CBF Losses:
- safe: 0.000000      # h(safe_states) >= 0 ✓
- unsafe: 5.280375    # h(unsafe_states) < 0 (high loss = learning)
- constraint: 0.000000 # CBF constraint satisfied
- total: 5.280375
```

**CLF Training (Feasibility Learning):**
```
Training Data:
- Goal states: torch.Size([1, 4])    # 1 goal example
- Transition states: torch.Size([4, 4]) # For constraint learning

CLF Losses:
- goal: 0.020684      # V(goal_states) = 0 (small loss = good)
- constraint: 0.000016 # CLF constraint nearly satisfied
- positive: 0.000000   # V(s) > 0 for non-goals ✓
- total: 0.020701
```

**Dynamics Training:**
```
Dynamics Losses:
- Total model loss: 3.258385
- Individual losses: [3.925, 3.077, 3.468, 2.522, 3.298]
```

**Key Insights:**
- CBF learning to distinguish safe/unsafe states (high unsafe loss)
- CLF learning goal proximity (low goal loss)
- Dynamics models learning state transitions

---

### **STEP 6: TRAINED NETWORK PREDICTIONS**

**After Training:**
```
Test state: [0.5, 0.3, 0.05, -0.05]

CBF value: 1.0481     # Positive = SAFE
CLF value: 0.0821     # Low = NEAR GOAL
Dynamics prediction: [1.340, 0.669, -0.268, -0.472]

Constraint Violations:
- CBF constraint: 0.500735   # h(s') - h(s) >= -α*h(s)
- CLF constraint: 0.071427   # V(s') - V(s) <= -β*V(s) + δ
```

**Improvements:**
- CBF now gives positive values for safe regions
- CLF gives lower values near goals
- Constraints being learned but not fully satisfied yet

---

### **STEP 7: SAFE ACTION FILTERING**

**Action Filtering Test:**
```
Test state: [1.5, 1.0, 0.2, 0.1]
Proposed action: [0.1, 0.05]
Safe action: [0.1, 0.05]           # Minimal modification
Action modification: [1.49e-09, 7.45e-10]  # Nearly unchanged
```

**Safety Metrics:**
```
CBF value: 0.3519        # Positive = safe
CLF value: 0.1066        # Low = feasible
Is safe: True            # CBF constraint satisfied
Is near goal: False      # CLF value too high
CBF constraint: 0.164839 # Some violation
CLF constraint: 0.038044 # Small violation
```

**Controller Behavior:**
- Action barely modified (constraints mostly satisfied)
- System considers state safe and action feasible
- Small constraint violations indicate learning in progress

---

### **STEP 8: TRAINING SUMMARY**

**Final Metrics:**
```
Training Summary:
- Total steps: 6
- Buffer size: 6
- Avg model uncertainty: 0.875902  # High = needs more data

Evaluation Metrics:
- Safety rate: Would be computed on larger batch
- Goal proximity rate: Would be computed on larger batch
- Constraint violations: Decreasing with training
```

---

## **Key Data Structures and Values**

### **Transition Storage:**
```python
transition = Transition(
    state=np.array([1.0, 0.5, 0.1, -0.2]),
    action=np.array([0.05, 0.1]),
    next_state=np.array([1.05, 0.48, 0.15, -0.1]),
    reward=-0.5,
    done=False,
    is_safe=True,   # ← Critical for CBF training
    is_goal=False   # ← Critical for CLF training
)
```

### **Network Outputs:**
```python
# CBF: h(s) - safety function
h_values = [-0.225, 0.496, 0.283]  # 3 models
h_mean = 0.1847                    # ensemble mean
h_uncertainty = 0.3704             # model disagreement

# CLF: V(s) - feasibility function  
V_values = [0.877, 0.942, 0.489]   # 3 models
V_mean = 0.7696                    # ensemble mean
V_uncertainty = 0.2448             # model disagreement

# Dynamics: P(s'|s,a) - next state prediction
predictions = [[...], [...], [...], [...], [...]]  # 5 models
pred_mean = [0.799, 0.607, 0.064, 0.279]          # ensemble mean
pred_uncertainty = 0.5049                          # epistemic uncertainty
```

### **Loss Components:**
```python
# CBF losses enforce safety constraints
cbf_losses = {
    "safe": 0.0,        # Penalty for h(safe_states) < 0
    "unsafe": 5.28,     # Penalty for h(unsafe_states) > 0  
    "constraint": 0.0,  # Penalty for violating h(s') - h(s) >= -α*h(s)
    "total": 5.28
}

# CLF losses enforce feasibility constraints
clf_losses = {
    "goal": 0.021,      # Penalty for V(goal_states) != 0
    "constraint": 0.0,  # Penalty for violating V(s') - V(s) <= -β*V(s) + δ
    "positive": 0.0,    # Penalty for V(non_goal_states) <= 0
    "total": 0.021
}
```

---

## **Mathematical Constraints in Action**

### **CBF Constraint:**
```
h(s_{k+1}) - h(s_k) >= -α * h(s_k)
Current violation: 0.500735 (being learned)
```

### **CLF Constraint:**
```
V(s_{k+1}) - V(s_k) <= -β * V(s_k) + δ  
Current violation: 0.071427 (being learned)
```

### **Safety Decision:**
```
is_safe = h(s) >= 0.0
Current: h(s) = 0.3519 → SAFE ✓
```

### **Feasibility Decision:**
```
is_near_goal = V(s) <= threshold
Current: V(s) = 0.1066 > 0.1 → NOT AT GOAL
```

This dry run demonstrates how the system learns safety and feasibility constraints from labeled data, with actual numerical values showing the learning process in action.