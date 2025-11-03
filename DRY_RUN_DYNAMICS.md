# End-to-End Dry Run: Dynamics in Navigation Task

## Initial State
```
Robot at: [1.0, 1.0, 0.0, 0.0]  # [x, y, vx, vy]
Goal at: [9.0, 7.0]
Obstacles at: [2.0, 2.0], [5.0, 3.0], [8.0, 1.5]
```

---

## STEP 1: Policy Proposes Action

**Input:**
- Current state: `s = [1.0, 1.0, 0.0, 0.0]`
- Subgoal from FSM: `g = [9.0, 7.0]`

**Policy Network:**
```python
# Policy: π_θ(s, g) → a
state_tensor = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
subgoal_tensor = torch.tensor([[9.0, 7.0]])

# Forward pass through neural network
action = policy(state_tensor, subgoal_tensor)
# Output: action = [0.3, 0.5]  (example)
```

**Exploration (10% chance):**
```python
if random() < 0.1:
    action = random_uniform(-0.5, 0.5, 2)  # Random exploration
```

**Output:** `a = [0.3, 0.5]` (acceleration in x, y)

---

## STEP 2: Real Environment Executes (Ground Truth)

**SimpleEnv.step(action):**
```python
# Current state
pos = [1.0, 1.0]
vel = [0.0, 0.0]
action = [0.3, 0.5]

# Physics: Euler integration with dt=0.1
vel_new = vel + action * 0.1
# vel_new = [0.0, 0.0] + [0.3, 0.5] * 0.1
# vel_new = [0.03, 0.05]

pos_new = pos + vel_new * 0.1
# pos_new = [1.0, 1.0] + [0.03, 0.05] * 0.1
# pos_new = [1.003, 1.005]

next_state = [1.003, 1.005, 0.03, 0.05]
```

**Collision Check:**
```python
for obstacle in obstacles:
    distance = ||pos_new - obstacle.center||
    if distance < obstacle.radius:
        collision = True

# [1.003, 1.005] vs [2.0, 2.0]: distance = 1.41 > 0.5 ✓ safe
# [1.003, 1.005] vs [5.0, 3.0]: distance = 4.47 > 0.4 ✓ safe
# [1.003, 1.005] vs [8.0, 1.5]: distance = 7.01 > 0.3 ✓ safe
collision = False
```

**Goal Check:**
```python
for goal in goals:
    distance = ||pos_new - goal.center||
    if distance < goal.radius:
        success = True

# [1.003, 1.005] vs [9.0, 7.0]: distance = 10.0 > 0.3 ✗
# [1.003, 1.005] vs [1.0, 7.0]: distance = 5.99 > 0.3 ✗
success = False
```

**Reward:**
```python
reward = -min_distance_to_goal
reward = -10.0  # Still far from goal
```

**Output:**
- `next_state = [1.003, 1.005, 0.03, 0.05]` ← GROUND TRUTH
- `reward = -10.0`
- `done = False`
- `info = {"collision": False, "success": False}`

---

## STEP 3: Store Transition in Buffer

```python
buffer.push(
    state = [1.0, 1.0, 0.0, 0.0],
    action = [0.3, 0.5],
    next_state = [1.003, 1.005, 0.03, 0.05],  # REAL from environment
    reward = -10.0,
    done = False
)
```

**Label for Learning:**
```python
if not collision:
    safe_states.append([1.0, 1.0, 0.0, 0.0])
if success:
    goal_states.append([1.003, 1.005, 0.03, 0.05])
```

---

## STEP 4: Update Dynamics Model (Every 5 Steps)

**Sample batch from buffer:**
```python
batch = buffer.sample(64)
# batch contains 64 transitions like:
# (s, a, s') where s' is REAL next state from environment
```

**Train dynamics to predict real transitions:**
```python
# For each transition in batch:
states = [[1.0, 1.0, 0.0, 0.0], ...]      # 64 states
actions = [[0.3, 0.5], ...]                # 64 actions
next_states_REAL = [[1.003, 1.005, 0.03, 0.05], ...]  # 64 REAL next states

# Forward pass through dynamics network
pred_next = dynamics(states, actions)
# pred_next = [[1.002, 1.004, 0.029, 0.048], ...]  (example prediction)

# Loss: How well does model predict REAL physics?
loss = mean((pred_next - next_states_REAL)²)
# loss = mean((1.002 - 1.003)² + (1.004 - 1.005)² + ...)
# loss = 0.0001  (small error = good prediction)

# Backprop and update
loss.backward()
optimizer.step()
```

**Result:** Dynamics model learns: `P̂_θ(s, a) ≈ real_physics(s, a)`

---

## STEP 5: Update Policy (Every Step)

**Sample batch from buffer:**
```python
batch = buffer.sample(64)
states = batch['states']  # 64 states
subgoals = [9.0, 7.0] repeated 64 times
```

**Policy gradient using LEARNED dynamics:**
```python
# For each state in batch:
s = [1.0, 1.0, 0.0, 0.0]
g = [9.0, 7.0]

# 1. Policy proposes action
a = policy(s, g)  # a = [0.3, 0.5]

# 2. Predict next state using LEARNED dynamics (not real env!)
ŝ' = dynamics(s, a)  # ŝ' = [1.002, 1.004, 0.029, 0.048]

# 3. Compute losses
# Task loss: How close to subgoal?
loss_task = ||ŝ'[:2] - g||²
loss_task = ||(1.002, 1.004) - (9.0, 7.0)||²
loss_task = 99.96  # Far from goal

# Safety penalty: Is predicted state safe?
h = cbf(ŝ')  # h = 0.8 (positive = safe)
loss_cbf = 0.5 * max(0, -h)²
loss_cbf = 0.5 * max(0, -0.8)² = 0  # Safe, no penalty

# Feasibility penalty: Is predicted state near goal?
V = clf(ŝ')  # V = 10.0 (high = far from goal)
loss_clf = 2.0 * max(0, V - 0.1)²
loss_clf = 2.0 * max(0, 10.0 - 0.1)²
loss_clf = 196.02  # Far from goal, high penalty

# Total loss
loss_policy = loss_task + loss_cbf + loss_clf
loss_policy = 99.96 + 0 + 196.02 = 296.0

# 4. Gradient flows through: policy → dynamics → loss
∂loss/∂policy_weights = ...
# Gradient tells policy: "Move toward goal, avoid obstacles"

# 5. Update policy
loss_policy.backward()
optimizer.step()
```

**Key Insight:** Policy uses LEARNED dynamics for gradient, not real environment!

---

## STEP 6: Update CBF (Every 10 Steps)

```python
# Use accumulated safe/unsafe states
safe_states = [[1.0, 1.0, 0.0, 0.0], ...]  # 100 safe states
unsafe_states = [[2.1, 2.1, 0.0, 0.0], ...]  # 10 unsafe states (near obstacles)

# Train CBF to separate them
h_safe = cbf(safe_states)  # Should be > 0
h_unsafe = cbf(unsafe_states)  # Should be < 0

loss_safe = mean(max(0, -h_safe)²)  # Penalize if h < 0 for safe
loss_unsafe = mean(max(0, h_unsafe)²)  # Penalize if h > 0 for unsafe

loss_cbf = loss_safe + loss_unsafe
loss_cbf.backward()
optimizer.step()
```

---

## STEP 7: Update CLF (Every 10 Steps)

```python
# Use accumulated goal states
goal_states = [[8.9, 6.9, 0.1, 0.1], ...]  # 5 states near goal

# Train CLF to be zero at goal
V_goal = clf(goal_states)  # Should be ≈ 0

loss_goal = mean(V_goal²)
loss_goal.backward()
optimizer.step()
```

---

## Summary: Two Different Dynamics Uses

### 1. EXECUTION (Real Environment)
```python
# Used for: Collecting ground truth data
next_state_REAL = env.step(action)
# Physics: pos_new = pos + (vel + action*dt)*dt
# This is the TRUE next state
```

### 2. TRAINING (Learned Model)
```python
# Used for: Computing policy gradients
next_state_PRED = dynamics(state, action)
# Neural network approximation of physics
# Allows gradient to flow: ∂loss/∂policy
```

### Why This Works (Model-Based RL)

1. **Dynamics learns from real data:**
   - Input: (s, a) from buffer
   - Target: s' from real environment
   - Loss: ||P̂(s,a) - s'||²

2. **Policy learns from learned dynamics:**
   - Input: (s, g)
   - Simulate: ŝ' = P̂(s, π(s,g))
   - Loss: ||ŝ' - g||² + penalties
   - Gradient flows through P̂ to π

3. **Co-training:**
   - Better dynamics → better policy gradients
   - Better policy → better data → better dynamics
   - Iterative improvement

### The Navigation Task

```
Start: [1.0, 1.0] → Goal: [9.0, 7.0]
Distance: 10 meters
Obstacles: 3 circular regions
Physics: Simple integrator (velocity + acceleration)
Success: Reach within 0.3m of goal without collision
```

After 100 episodes, policy learns to:
1. Move toward goal (CLF gradient)
2. Avoid obstacles (CBF penalty)
3. Use smooth control (learned dynamics)
