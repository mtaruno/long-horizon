"""
Demonstration of how PD controllers work WITH CBF-CLF, not against them.
"""

import numpy as np
import torch
from src import create_trainer, Transition

def demonstrate_pd_cbf_clf_cooperation():
    """Show how PD and CBF-CLF work together, not in conflict."""
    
    print("=== PD CONTROLLER + CBF-CLF COOPERATION ===\n")
    
    print("KEY INSIGHT: They operate at DIFFERENT LEVELS")
    print("- PD Controller: Converts GOALS → ACTIONS")
    print("- CBF-CLF: Filters ACTIONS → SAFE ACTIONS")
    print()
    
    # Initialize safety trainer
    trainer = create_trainer(state_dim=4, action_dim=2, device="cpu")
    
    # Add some training data to make CBF-CLF meaningful
    safe_transitions = [
        ([1.0, 1.0, 0.1, 0.1], [0.1, 0.1], [1.01, 1.01, 0.2, 0.2], True, False),
        ([0.5, 0.5, 0.05, 0.05], [0.05, 0.05], [0.505, 0.505, 0.1, 0.1], True, False),
        ([2.0, 1.5, 0.2, 0.1], [0.0, 0.0], [2.02, 1.51, 0.2, 0.1], True, False),
    ]
    
    unsafe_transitions = [
        ([2.9, 2.8, 0.5, 0.4], [0.2, 0.2], [3.0, 2.9, 0.7, 0.6], False, False),  # Near boundary
    ]
    
    goal_transitions = [
        ([0.05, 0.05, 0.01, 0.01], [0.0, 0.0], [0.05, 0.05, 0.01, 0.01], True, True),  # At goal
    ]
    
    # Add training data
    for s, a, ns, is_safe, is_goal in safe_transitions + unsafe_transitions + goal_transitions:
        transition = Transition(
            state=np.array(s), action=np.array(a), next_state=np.array(ns),
            reward=0.0, done=is_goal, is_safe=is_safe, is_goal=is_goal
        )
        trainer.add_transition(transition)
    
    print("Added training data for CBF-CLF learning...")
    print()
    
    # Test scenarios
    scenarios = [
        {
            "name": "SAFE SCENARIO - PD and CBF-CLF Agree",
            "current_state": np.array([1.0, 1.0, 0.1, 0.1]),
            "target_position": np.array([1.5, 1.2]),
            "target_velocity": np.array([0.2, 0.1])
        },
        {
            "name": "UNSAFE SCENARIO - CBF-CLF Overrides PD",
            "current_state": np.array([2.8, 2.7, 0.4, 0.3]),
            "target_position": np.array([3.2, 3.0]),  # Aggressive target
            "target_velocity": np.array([0.6, 0.5])   # High velocity
        },
        {
            "name": "GOAL SCENARIO - CLF Guides PD",
            "current_state": np.array([0.2, 0.15, 0.05, 0.03]),
            "target_position": np.array([0.0, 0.0]),  # Go to origin (goal)
            "target_velocity": np.array([0.0, 0.0])
        }
    ]
    
    for scenario in scenarios:
        print(f"SCENARIO: {scenario['name']}")
        print("-" * 60)
        
        state = scenario["current_state"]
        target_pos = scenario["target_position"]
        target_vel = scenario["target_velocity"]
        
        print(f"Current state: [x={state[0]:.2f}, y={state[1]:.2f}, vx={state[2]:.2f}, vy={state[3]:.2f}]")
        print(f"Target position: {target_pos}")
        print(f"Target velocity: {target_vel}")
        
        # STEP 1: PD Controller (Goal → Action)
        current_pos = state[:2]
        current_vel = state[2:]
        
        pos_error = target_pos - current_pos
        vel_error = target_vel - current_vel
        
        Kp, Kd = 5.0, 2.0
        pd_action = Kp * pos_error + Kd * vel_error
        
        print(f"\nSTEP 1 - PD Controller:")
        print(f"  Position error: {pos_error}")
        print(f"  Velocity error: {vel_error}")
        print(f"  PD action: {pd_action}")
        
        # STEP 2: CBF-CLF Safety Filter (Action → Safe Action)
        safe_action = trainer.get_safe_action(state, pd_action)
        
        print(f"\nSTEP 2 - CBF-CLF Safety Filter:")
        print(f"  Proposed action: {pd_action}")
        print(f"  Safe action: {safe_action.numpy()}")
        
        action_modified = not np.allclose(pd_action, safe_action.numpy(), atol=1e-6)
        print(f"  Action modified: {action_modified}")
        
        if action_modified:
            modification = safe_action.numpy() - pd_action
            print(f"  Modification: {modification}")
        
        # STEP 3: Safety Analysis
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        pd_action_tensor = torch.FloatTensor(pd_action).unsqueeze(0)
        
        metrics = trainer.controller.get_safety_feasibility_metrics(
            state_tensor, pd_action_tensor, trainer.dynamics_ensemble
        )
        
        print(f"\nSTEP 3 - Safety Analysis:")
        print(f"  CBF value: {metrics['cbf_values'].item():.4f} ({'SAFE' if metrics['is_safe'].item() else 'UNSAFE'})")
        print(f"  CLF value: {metrics['clf_values'].item():.4f} ({'NEAR GOAL' if metrics['is_near_goal'].item() else 'FAR FROM GOAL'})")
        print(f"  CBF constraint violation: {metrics['cbf_constraints'].item():.6f}")
        print(f"  CLF constraint violation: {metrics['clf_constraints'].item():.6f}")
        
        print(f"\n" + "="*70 + "\n")

def explain_complementary_roles():
    """Explain how PD and CBF-CLF have complementary, not conflicting roles."""
    
    print("=== COMPLEMENTARY ROLES: PD vs CBF-CLF ===\n")
    
    print("┌─────────────────────────────────────────────────────────────┐")
    print("│                    CONTROL HIERARCHY                        │")
    print("├─────────────────────────────────────────────────────────────┤")
    print("│  HIGH-LEVEL PLANNER                                         │")
    print("│  ├─ FSM/LLM: \"Go to corner\"                               │")
    print("│  └─ Output: target_position=[5,5], target_velocity=[0.5,0.5]│")
    print("├─────────────────────────────────────────────────────────────┤")
    print("│  PD CONTROLLER (Performance Layer)                         │")
    print("│  ├─ Role: Convert goals → actions                          │")
    print("│  ├─ Input: target_position, target_velocity                 │")
    print("│  ├─ Logic: action = Kp*pos_error + Kd*vel_error           │")
    print("│  └─ Output: proposed_action=[1.2, 0.8]                    │")
    print("├─────────────────────────────────────────────────────────────┤")
    print("│  CBF-CLF FRAMEWORK (Safety Layer)                          │")
    print("│  ├─ Role: Ensure actions are safe & feasible              │")
    print("│  ├─ Input: current_state, proposed_action                   │")
    print("│  ├─ Logic: Check h(s') ≥ -α*h(s), V(s') ≤ -β*V(s)+δ      │")
    print("│  └─ Output: safe_action=[1.1, 0.7] (modified if needed)   │")
    print("├─────────────────────────────────────────────────────────────┤")
    print("│  ROBOT EXECUTION                                            │")
    print("│  └─ Execute: safe_action                                    │")
    print("└─────────────────────────────────────────────────────────────┘")
    
    print("\nKEY INSIGHTS:")
    print("1. DIFFERENT OBJECTIVES:")
    print("   - PD Controller: Achieve performance goals")
    print("   - CBF-CLF: Maintain safety constraints")
    
    print("\n2. SEQUENTIAL OPERATION:")
    print("   - PD runs first: goal → action")
    print("   - CBF-CLF runs second: action → safe_action")
    
    print("\n3. COMPLEMENTARY STRENGTHS:")
    print("   - PD: Fast, intuitive, goal-oriented")
    print("   - CBF-CLF: Mathematically rigorous, safety-guaranteed")
    
    print("\n4. NO CONFLICT:")
    print("   - PD doesn't know about safety constraints")
    print("   - CBF-CLF doesn't care about performance goals")
    print("   - They operate in different domains")

def demonstrate_when_cbf_clf_intervenes():
    """Show specific cases where CBF-CLF overrides PD controller."""
    
    print("\n=== WHEN CBF-CLF INTERVENES ===\n")
    
    intervention_cases = [
        {
            "name": "Safety Intervention",
            "description": "PD wants to go fast, but CBF says it's unsafe",
            "state": [2.5, 2.3, 0.3, 0.2],
            "pd_target": [3.0, 2.8],  # Near boundary
            "expected": "CBF reduces acceleration to maintain safety"
        },
        {
            "name": "Feasibility Intervention", 
            "description": "PD overshoots goal, CLF corrects trajectory",
            "state": [0.3, 0.2, 0.1, 0.05],
            "pd_target": [0.8, 0.6],  # Away from goal
            "expected": "CLF redirects toward goal region"
        },
        {
            "name": "No Intervention",
            "description": "PD action is already safe and feasible",
            "state": [1.0, 1.0, 0.1, 0.1],
            "pd_target": [1.2, 1.1],  # Safe region
            "expected": "CBF-CLF passes through unchanged"
        }
    ]
    
    print("INTERVENTION SCENARIOS:")
    print("-" * 50)
    
    for case in intervention_cases:
        print(f"\n{case['name']}:")
        print(f"  Description: {case['description']}")
        print(f"  State: {case['state']}")
        print(f"  PD Target: {case['pd_target']}")
        print(f"  Expected: {case['expected']}")

def show_mathematical_compatibility():
    """Show that PD and CBF-CLF are mathematically compatible."""
    
    print("\n=== MATHEMATICAL COMPATIBILITY ===\n")
    
    print("PD CONTROLLER EQUATION:")
    print("  u_pd = Kp * (x_target - x_current) + Kd * (v_target - v_current)")
    print("  Domain: Goal achievement")
    print("  Constraint: None (can propose any action)")
    
    print("\nCBF CONSTRAINT:")
    print("  h(x_next) - h(x_current) ≥ -α * h(x_current)")
    print("  Domain: Safety maintenance")
    print("  Constraint: Actions must preserve safety")
    
    print("\nCLF CONSTRAINT:")
    print("  V(x_next) - V(x_current) ≤ -β * V(x_current) + δ")
    print("  Domain: Goal convergence")
    print("  Constraint: Actions must approach goal")
    
    print("\nCOMBINED SYSTEM:")
    print("  u_safe = argmin ||u - u_pd||²")
    print("  subject to:")
    print("    h(f(x,u)) - h(x) ≥ -α * h(x)  (CBF)")
    print("    V(f(x,u)) - V(x) ≤ -β * V(x) + δ  (CLF)")
    print()
    print("  Translation: Find action closest to PD's proposal")
    print("               that satisfies safety constraints")
    
    print("\nWHY THIS WORKS:")
    print("  1. PD provides 'desired' behavior")
    print("  2. CBF-CLF provides 'allowable' region")
    print("  3. Optimization finds best compromise")
    print("  4. Result: Safe action close to desired action")

if __name__ == "__main__":
    demonstrate_pd_cbf_clf_cooperation()
    explain_complementary_roles()
    demonstrate_when_cbf_clf_intervenes()
    show_mathematical_compatibility()