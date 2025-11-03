# Training Process Explained

## Q1: What is Adaptive Model Learning For?

**Purpose**: Learn the dynamics P̂_θ(s, a) → s' to predict what happens when you take an action.

**Why needed**:
1. **CBF/CLF constraints require predictions**: To check if action is safe, we need to know where it leads
2. **Policy training uses model**: L_actor uses P̂_θ(s, a) to predict next state
3. **Uncertainty quantification**: Ensemble tells us when predictions are unreliable

**Ground truth**: Real transitions (s, a, s') collected from environment

```python
# Collect real data
next_state_real = env.step(action)  # Ground truth

# Train model to predict it
next_state_pred = dynamics(state, action)
loss = ||next_state_pred - next_state_real||²
```

## Q2: Data Collection Order

**You're right to question this!** The training has a **bootstrapping problem**:

### Phase 1: Initial Data Collection (Random/Heuristic Policy)

```python
# Start with random or simple policy
for step in range(initial_steps):
    action = random_action()  # or simple_heuristic(state)
    next_state, reward, done, info = env.step(action)
    
    # Store real transition
    buffer.push(state, action, next_state, reward, done)
    
    # Label safety/goal from environment
    is_safe = not info['collision']
    is_goal = info['success']
```

### Phase 2: Online Learning (Iterative Improvement)

```python
# Now we have data, start learning
for episode in range(num_episodes):
    for step in range(max_steps):
        # 1. Policy uses current (imperfect) models
        action = policy(state, subgoal)
        
        # 2. Execute in REAL environment (ground truth)
        next_state_real, reward, done, info = env.step(action)
        
        # 3. Store real transition
        buffer.push(state, action, next_state_real, ...)
        
        # 4. Update models with real data
        if step % update_freq == 0:
            # Dynamics: learn to predict real transitions
            batch = buffer.sample(64)
            pred = dynamics(batch['states'], batch['actions'])
            loss_dyn = ||pred - batch['next_states']||²  # Real next_states!
            
            # CBF: learn from real safe/unsafe states
            loss_cbf = cbf_loss(real_safe_states, real_unsafe_states, ...)
            
            # CLF: learn from real goal states
            loss_clf = clf_loss(real_goal_states, ...)
            
            # Policy: use learned models for gradient
            pred_next = dynamics(state, policy(state, subgoal))
            loss_policy = distance(pred_next, subgoal) + cbf_penalty + clf_penalty
```

## Q3: Policy Training - Loss Function & Ground Truth

### The Key Insight: Model-Based Policy Gradient

**Policy does NOT need ground truth actions!** It learns by:
1. Using learned dynamics to predict outcomes
2. Optimizing to reach subgoals while satisfying constraints

### Policy Loss Breakdown

```python
# Given: state s, subgoal g
action = policy(state, subgoal)  # Policy proposes action

# Use LEARNED dynamics (not ground truth)
predicted_next_state = dynamics(state, action)

# Loss components:
# 1. Subgoal distance (task objective)
loss_task = ||predicted_next_state[:2] - subgoal||²

# 2. CBF penalty (safety)
h_value = cbf(predicted_next_state)
loss_cbf = λ_cbf * [max(0, -h_value)]²

# 3. CLF penalty (feasibility)
V_value = clf(predicted_next_state)
loss_clf = λ_clf * [max(0, V_value - ε)]²

# Total
loss_policy = loss_task + loss_cbf + loss_clf
```

**Ground truth**: The subgoal from FSM (where we want to go)

### Why This Works

The policy learns through **differentiable simulation**:
- Dynamics model approximates real environment
- Policy gradient flows through: policy → dynamics → loss
- As dynamics improves (from real data), policy improves

## Complete Data Flow Diagram

```
┌─────────────────────────────────────────────────────────┐
│ REAL ENVIRONMENT (Source of Ground Truth)               │
│                                                          │
│  state, action → [Physics] → next_state_REAL           │
│                               ↓                          │
│                          is_safe? is_goal?              │
└─────────────────────────────────────────────────────────┘
                               ↓
                    Store in Replay Buffer
                               ↓
┌─────────────────────────────────────────────────────────┐
│ LEARNING (Uses Real Data as Ground Truth)               │
│                                                          │
│ 1. Dynamics Learning:                                   │
│    pred_next = P̂_θ(s, a)                               │
│    loss = ||pred_next - next_state_REAL||²             │
│    Ground truth: next_state_REAL from buffer            │
│                                                          │
│ 2. CBF Learning:                                        │
│    h(s) for s in {safe_states_REAL, unsafe_states_REAL}│
│    Ground truth: safety labels from environment         │
│                                                          │
│ 3. CLF Learning:                                        │
│    V(s) for s in {goal_states_REAL, trajectory_states} │
│    Ground truth: goal labels from environment           │
│                                                          │
│ 4. Policy Learning:                                     │
│    a = π(s, g)                                          │
│    pred_next = P̂_θ(s, a)  ← Uses learned dynamics     │
│    loss = ||pred_next - g||² + penalties                │
│    Ground truth: subgoal g from FSM                     │
└─────────────────────────────────────────────────────────┘
```

## Concrete Example

### Step 1: Collect Initial Data (100 steps with random policy)

```python
for i in range(100):
    action = np.random.uniform(-1, 1, 2)
    next_state, _, _, info = env.step(action)
    buffer.push(state, action, next_state)
    
    if not info['collision']:
        safe_states.append(state)
    else:
        unsafe_states.append(state)
```

**Ground truth collected**: 100 real transitions (s, a, s')

### Step 2: Train Dynamics

```python
batch = buffer.sample(64)
pred = dynamics(batch['states'], batch['actions'])
loss = MSE(pred, batch['next_states'])  # next_states are REAL
```

**Ground truth used**: Real next_states from environment

### Step 3: Train CBF

```python
safe = torch.tensor(safe_states)
unsafe = torch.tensor(unsafe_states)

h_safe = cbf(safe)
h_unsafe = cbf(unsafe)

loss = mean([max(0, -h_safe)]²) + mean([max(0, h_unsafe)]²)
```

**Ground truth used**: Real safety labels from collisions

### Step 4: Train Policy

```python
state = torch.tensor([1.0, 2.0, 0.1, 0.2])
subgoal = torch.tensor([5.0, 5.0])  # From FSM

action = policy(state, subgoal)
pred_next = dynamics(state, action)  # Use learned model

loss = ||pred_next[:2] - subgoal||²  # Distance to subgoal
```

**Ground truth used**: Subgoal from FSM (not action!)

### Step 5: Collect More Data with Improved Policy

```python
action = policy(state, subgoal)  # Better than random now
next_state_real, _, _, info = env.step(action)
buffer.push(state, action, next_state_real)  # New ground truth!
```

**Ground truth collected**: New real transitions with better policy

## Key Takeaways

1. **Dynamics ground truth**: Real next states from environment
2. **CBF ground truth**: Real safety labels (collision detection)
3. **CLF ground truth**: Real goal labels (success detection)
4. **Policy ground truth**: Subgoals from FSM (NOT actions!)

5. **Bootstrapping**: Start with random data, iteratively improve

6. **Model-based RL**: Policy uses learned dynamics, not real environment, for gradients

7. **Online learning**: Continuously collect real data and update models
