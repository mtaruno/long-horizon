"""
Explanation of the initial dataset and labeling process for CBF-CLF training.
"""

import numpy as np
from src import Transition

def explain_initial_dataset():
    """Explain what the initial training dataset looks like."""
    
    print("=== INITIAL DATASET STRUCTURE ===\n")
    
    print("DATASET = Collection of LABELED TRANSITIONS")
    print()
    
    print("Each transition contains:")
    print("┌─────────────────────────────────────────────────────────┐")
    print("│ Transition(                                             │")
    print("│   state=[x, y, vx, vy],        # Current robot state    │")
    print("│   action=[ax, ay],             # Action taken           │")
    print("│   next_state=[x', y', vx', vy'], # Resulting state     │")
    print("│   reward=float,                # Optional reward       │")
    print("│   done=bool,                   # Episode ended?        │")
    print("│   is_safe=bool,                # MANUAL LABEL: Safe?   │")
    print("│   is_goal=bool                 # MANUAL LABEL: Goal?   │")
    print("│ )                                                       │")
    print("└─────────────────────────────────────────────────────────┘")
    print()

def show_example_transitions():
    """Show concrete examples of labeled transitions."""
    
    print("=== EXAMPLE LABELED TRANSITIONS ===\n")
    
    # Example 1: Safe transition
    print("EXAMPLE 1: Safe Navigation")
    safe_transition = Transition(
        state=np.array([1.0, 1.0, 0.1, 0.1]),
        action=np.array([0.2, 0.1]),
        next_state=np.array([1.02, 1.01, 0.12, 0.11]),
        reward=-1.41,  # Distance to origin
        done=False,
        is_safe=True,   # MANUAL LABEL: No collision
        is_goal=False   # MANUAL LABEL: Not at goal
    )
    
    print(f"State: {safe_transition.state} (robot in open space)")
    print(f"Action: {safe_transition.action} (gentle acceleration)")
    print(f"Next state: {safe_transition.next_state} (moved safely)")
    print(f"is_safe: {safe_transition.is_safe} ← HUMAN LABELED")
    print(f"is_goal: {safe_transition.is_goal} ← HUMAN LABELED")
    print("Reasoning: Robot moved in open space, no obstacles hit")
    print()
    
    # Example 2: Unsafe transition
    print("EXAMPLE 2: Collision/Unsafe")
    unsafe_transition = Transition(
        state=np.array([2.9, 2.8, 0.3, 0.2]),
        action=np.array([0.5, 0.4]),
        next_state=np.array([3.0, 2.9, 0.35, 0.24]),
        reward=-10.0,  # Penalty for collision
        done=True,
        is_safe=False,  # MANUAL LABEL: Collision occurred!
        is_goal=False
    )
    
    print(f"State: {unsafe_transition.state} (near boundary)")
    print(f"Action: {unsafe_transition.action} (toward boundary)")
    print(f"Next state: {unsafe_transition.next_state} (hit boundary)")
    print(f"is_safe: {unsafe_transition.is_safe} ← HUMAN LABELED")
    print(f"is_goal: {unsafe_transition.is_goal} ← HUMAN LABELED")
    print("Reasoning: Robot hit wall/obstacle at position [3.0, 2.9]")
    print()
    
    # Example 3: Goal reached
    print("EXAMPLE 3: Goal Achievement")
    goal_transition = Transition(
        state=np.array([0.1, 0.05, 0.02, 0.01]),
        action=np.array([-0.02, -0.01]),
        next_state=np.array([0.08, 0.04, 0.0, 0.0]),
        reward=100.0,  # Big reward for reaching goal
        done=True,
        is_safe=True,
        is_goal=True    # MANUAL LABEL: Reached goal!
    )
    
    print(f"State: {goal_transition.state} (very close to origin)")
    print(f"Action: {goal_transition.action} (small braking)")
    print(f"Next state: {goal_transition.next_state} (at origin)")
    print(f"is_safe: {goal_transition.is_safe} ← HUMAN LABELED")
    print(f"is_goal: {goal_transition.is_goal} ← HUMAN LABELED")
    print("Reasoning: Robot reached target location [0, 0]")
    print()

def explain_labeling_process():
    """Explain how manual labeling is done."""
    
    print("=== MANUAL LABELING PROCESS ===\n")
    
    print("HOW ARE LABELS CREATED?")
    print()
    
    print("METHOD 1: Simulation-Based Labeling")
    print("┌─────────────────────────────────────────────────────────┐")
    print("│ 1. Run robot in simulator (Isaac Gym, Gazebo, etc.)    │")
    print("│ 2. Simulator detects collisions automatically          │")
    print("│ 3. is_safe = not collision_detected                    │")
    print("│ 4. is_goal = distance_to_target < threshold            │")
    print("│ 5. Collect thousands of transitions automatically      │")
    print("└─────────────────────────────────────────────────────────┘")
    print()
    
    print("METHOD 2: Real Robot Data Collection")
    print("┌─────────────────────────────────────────────────────────┐")
    print("│ 1. Human operator drives robot around                  │")
    print("│ 2. Record all state-action-next_state transitions      │")
    print("│ 3. Human labels each transition afterward:             │")
    print("│    - 'Was this safe?' (no collision/danger)           │")
    print("│    - 'Did this reach goal?' (task completed)          │")
    print("│ 4. Time-intensive but high-quality labels             │")
    print("└─────────────────────────────────────────────────────────┘")
    print()
    
    print("METHOD 3: Domain Knowledge Rules")
    print("┌─────────────────────────────────────────────────────────┐")
    print("│ 1. Define safety rules: 'safe if distance > 0.5m'     │")
    print("│ 2. Define goal rules: 'goal if ||pos - target|| < 0.1' │")
    print("│ 3. Apply rules automatically to label data             │")
    print("│ 4. Fast but may miss edge cases                       │")
    print("└─────────────────────────────────────────────────────────┘")
    print()

def show_dataset_creation_example():
    """Show how a dataset is created in practice."""
    
    print("=== DATASET CREATION EXAMPLE ===\n")
    
    print("SCENARIO: Training a warehouse robot")
    print()
    
    print("STEP 1: Define Environment")
    print("- Warehouse: 10m × 8m rectangle")
    print("- Obstacles: Shelves, walls, other robots")
    print("- Goal: Delivery locations")
    print("- Safe region: Open floor space")
    print()
    
    print("STEP 2: Data Collection")
    print("Method: Isaac Gym simulation")
    
    # Simulate data collection
    transitions = []
    
    print("\nCollecting transitions...")
    
    # Safe transitions (most common)
    for i in range(3):
        state = np.random.uniform([0.5, 0.5, -0.2, -0.2], [8.5, 6.5, 0.2, 0.2])
        action = np.random.uniform([-0.5, -0.5], [0.5, 0.5])
        next_state = state + np.concatenate([state[2:], action]) * 0.1
        
        # Check safety (simple rule: inside bounds)
        is_safe = (0 < next_state[0] < 9) and (0 < next_state[1] < 7)
        is_goal = np.linalg.norm(next_state[:2] - np.array([8.0, 6.0])) < 0.2
        
        transition = Transition(
            state=state, action=action, next_state=next_state,
            reward=0.0, done=is_goal, is_safe=is_safe, is_goal=is_goal
        )
        transitions.append(transition)
        
        print(f"Transition {i+1}: safe={is_safe}, goal={is_goal}")
    
    print(f"\nCollected {len(transitions)} transitions")
    print()
    
    print("STEP 3: Label Statistics")
    safe_count = sum(1 for t in transitions if t.is_safe)
    goal_count = sum(1 for t in transitions if t.is_goal)
    
    print(f"Safe transitions: {safe_count}/{len(transitions)} ({safe_count/len(transitions)*100:.1f}%)")
    print(f"Goal transitions: {goal_count}/{len(transitions)} ({goal_count/len(transitions)*100:.1f}%)")
    print(f"Unsafe transitions: {len(transitions)-safe_count}/{len(transitions)}")
    print()

def explain_what_networks_learn():
    """Explain what each network learns from the labeled data."""
    
    print("=== WHAT NETWORKS LEARN FROM LABELS ===\n")
    
    print("CBF NETWORK learns from is_safe labels:")
    print("┌─────────────────────────────────────────────────────────┐")
    print("│ Input: All states from transitions                      │")
    print("│ Target: h(state) > 0 if is_safe=True                  │")
    print("│         h(state) < 0 if is_safe=False                 │")
    print("│                                                         │")
    print("│ Result: h(s) becomes a 'safety detector'               │")
    print("│         Positive values = safe regions                  │")
    print("│         Negative values = unsafe regions                │")
    print("└─────────────────────────────────────────────────────────┘")
    print()
    
    print("CLF NETWORK learns from is_goal labels:")
    print("┌─────────────────────────────────────────────────────────┐")
    print("│ Input: All states from transitions                      │")
    print("│ Target: V(state) = 0 if is_goal=True                  │")
    print("│         V(state) > 0 if is_goal=False                 │")
    print("│                                                         │")
    print("│ Result: V(s) becomes a 'distance to goal'              │")
    print("│         Zero values = at goal                           │")
    print("│         Higher values = farther from goal               │")
    print("└─────────────────────────────────────────────────────────┘")
    print()
    
    print("DYNAMICS NETWORK learns from state transitions:")
    print("┌─────────────────────────────────────────────────────────┐")
    print("│ Input: (state, action) pairs                           │")
    print("│ Target: next_state (actual outcome)                    │")
    print("│                                                         │")
    print("│ Result: f(s,a) predicts what happens next              │")
    print("│         Used by CBF-CLF to check constraints           │")
    print("└─────────────────────────────────────────────────────────┘")
    print()

def show_data_requirements():
    """Show typical data requirements for training."""
    
    print("=== DATA REQUIREMENTS ===\n")
    
    print("TYPICAL DATASET SIZES:")
    print("- Small robot (2D): 1,000-10,000 transitions")
    print("- Complex robot (high-dim): 100,000+ transitions")
    print("- Continuous learning: Add data during operation")
    print()
    
    print("LABEL DISTRIBUTION (typical):")
    print("- Safe transitions: 80-95% (most robot operation)")
    print("- Unsafe transitions: 5-15% (collisions, boundaries)")
    print("- Goal transitions: 1-5% (successful task completion)")
    print()
    
    print("DATA QUALITY REQUIREMENTS:")
    print("✓ Diverse states (cover entire workspace)")
    print("✓ Diverse actions (different behaviors)")
    print("✓ Accurate labels (critical for safety)")
    print("✓ Balanced dataset (enough unsafe examples)")
    print("✓ Representative scenarios (real conditions)")
    print()

def compare_labeling_approaches():
    """Compare different approaches to creating labeled data."""
    
    print("=== LABELING APPROACHES COMPARISON ===\n")
    
    approaches = [
        {
            "name": "Simulation Auto-Labeling",
            "pros": ["Fast", "Scalable", "Consistent", "Safe to collect"],
            "cons": ["Sim-to-real gap", "May miss edge cases"],
            "best_for": "Initial training, rapid prototyping"
        },
        {
            "name": "Human Expert Labeling", 
            "pros": ["High quality", "Captures nuance", "Real-world grounded"],
            "cons": ["Slow", "Expensive", "Subjective", "Limited scale"],
            "best_for": "Critical applications, validation data"
        },
        {
            "name": "Rule-Based Labeling",
            "pros": ["Fast", "Deterministic", "Explainable"],
            "cons": ["Rigid", "May miss complexity", "Hard to generalize"],
            "best_for": "Well-defined environments, initial datasets"
        },
        {
            "name": "Active Learning",
            "pros": ["Efficient", "Focuses on hard cases", "Adaptive"],
            "cons": ["Complex", "Requires initial model", "Iterative"],
            "best_for": "Improving existing models, edge case discovery"
        }
    ]
    
    for approach in approaches:
        print(f"{approach['name']}:")
        print(f"  Pros: {', '.join(approach['pros'])}")
        print(f"  Cons: {', '.join(approach['cons'])}")
        print(f"  Best for: {approach['best_for']}")
        print()

if __name__ == "__main__":
    explain_initial_dataset()
    show_example_transitions()
    explain_labeling_process()
    show_dataset_creation_example()
    explain_what_networks_learn()
    show_data_requirements()
    compare_labeling_approaches()