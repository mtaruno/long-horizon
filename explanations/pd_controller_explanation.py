"""
PD Controller Explanation with Visual Examples
"""

import numpy as np

def explain_pd_controller():
    """Explain PD controller with concrete examples."""
    
    print("=== PD CONTROLLER EXPLANATION ===\n")
    
    print("PD = Proportional + Derivative")
    print("Formula: action = Kp * error + Kd * error_rate")
    print()
    
    # Example scenario
    print("SCENARIO: Robot Navigation")
    print("-" * 30)
    
    current_pos = np.array([1.0, 1.0])      # Where robot is now
    target_pos = np.array([3.0, 2.0])       # Where robot wants to go
    current_vel = np.array([0.1, 0.1])      # Current velocity
    target_vel = np.array([0.5, 0.3])       # Desired velocity
    
    print(f"Current position: {current_pos}")
    print(f"Target position:  {target_pos}")
    print(f"Current velocity: {current_vel}")
    print(f"Target velocity:  {target_vel}")
    
    # Calculate errors
    position_error = target_pos - current_pos
    velocity_error = target_vel - current_vel
    
    print(f"\nErrors:")
    print(f"Position error: {position_error}")
    print(f"Velocity error: {velocity_error}")
    
    # PD controller gains
    Kp = 5.0  # Proportional gain (how aggressively to correct position)
    Kd = 2.0  # Derivative gain (how much to consider velocity)
    
    print(f"\nPD Gains:")
    print(f"Kp (proportional): {Kp}")
    print(f"Kd (derivative):   {Kd}")
    
    # Calculate control action (acceleration)
    proportional_term = Kp * position_error
    derivative_term = Kd * velocity_error
    acceleration = proportional_term + derivative_term
    
    print(f"\nPD Controller Calculation:")
    print(f"Proportional term: Kp * pos_error = {Kp} * {position_error} = {proportional_term}")
    print(f"Derivative term:   Kd * vel_error = {Kd} * {velocity_error} = {derivative_term}")
    print(f"Total acceleration: {acceleration}")
    
    return acceleration

def demonstrate_pd_behavior():
    """Show how PD controller behaves in different scenarios."""
    
    print("\n=== PD CONTROLLER BEHAVIOR ===\n")
    
    scenarios = [
        {
            "name": "Far from target, slow velocity",
            "pos_error": np.array([2.0, 1.0]),
            "vel_error": np.array([0.4, 0.2])
        },
        {
            "name": "Close to target, fast velocity", 
            "pos_error": np.array([0.1, 0.1]),
            "vel_error": np.array([-0.3, -0.2])
        },
        {
            "name": "At target, but moving too fast",
            "pos_error": np.array([0.0, 0.0]),
            "vel_error": np.array([-0.5, -0.3])
        },
        {
            "name": "Overshoot - past target",
            "pos_error": np.array([-0.5, -0.2]),
            "vel_error": np.array([-0.2, -0.1])
        }
    ]
    
    Kp, Kd = 5.0, 2.0
    
    for scenario in scenarios:
        print(f"SCENARIO: {scenario['name']}")
        pos_err = scenario['pos_error']
        vel_err = scenario['vel_error']
        
        P_term = Kp * pos_err
        D_term = Kd * vel_err
        action = P_term + D_term
        
        print(f"  Position error: {pos_err}")
        print(f"  Velocity error: {vel_err}")
        print(f"  P term: {P_term}")
        print(f"  D term: {D_term}")
        print(f"  Action: {action}")
        print(f"  Interpretation: {interpret_action(action)}")
        print()

def interpret_action(action):
    """Interpret what the acceleration command means."""
    magnitude = np.linalg.norm(action)
    
    if magnitude < 0.1:
        return "Gentle adjustment"
    elif magnitude < 1.0:
        return "Moderate acceleration"
    elif magnitude < 2.0:
        return "Strong acceleration"
    else:
        return "Maximum effort"

def compare_pd_gains():
    """Show how different PD gains affect behavior."""
    
    print("=== EFFECT OF DIFFERENT PD GAINS ===\n")
    
    pos_error = np.array([1.0, 0.5])  # Fixed scenario
    vel_error = np.array([0.2, 0.1])
    
    gain_sets = [
        {"name": "Conservative", "Kp": 1.0, "Kd": 0.5},
        {"name": "Moderate", "Kp": 3.0, "Kd": 1.5},
        {"name": "Aggressive", "Kp": 8.0, "Kd": 3.0},
        {"name": "High Damping", "Kp": 3.0, "Kd": 5.0}
    ]
    
    print(f"Fixed scenario: pos_error={pos_error}, vel_error={vel_error}")
    print()
    
    for gains in gain_sets:
        Kp, Kd = gains["Kp"], gains["Kd"]
        action = Kp * pos_error + Kd * vel_error
        
        print(f"{gains['name']:12} (Kp={Kp:3.1f}, Kd={Kd:3.1f}): action={action} → {interpret_action(action)}")

def pd_vs_other_controllers():
    """Compare PD with other control strategies."""
    
    print("\n=== PD vs OTHER CONTROLLERS ===\n")
    
    pos_error = np.array([1.5, 1.0])
    vel_error = np.array([0.3, 0.2])
    
    print(f"Scenario: pos_error={pos_error}, vel_error={vel_error}")
    print()
    
    # P-only controller (no derivative term)
    Kp = 5.0
    p_action = Kp * pos_error
    print(f"P-only controller:  action = {p_action}")
    print(f"  Problem: No velocity consideration → oscillation/overshoot")
    
    # PD controller
    Kd = 2.0
    pd_action = Kp * pos_error + Kd * vel_error
    print(f"PD controller:      action = {pd_action}")
    print(f"  Better: Considers velocity → smoother approach")
    
    # Bang-bang controller
    bang_action = np.sign(pos_error) * 2.0  # Max acceleration toward target
    print(f"Bang-bang:          action = {bang_action}")
    print(f"  Problem: Jerky motion, hard on actuators")

def real_world_pd_example():
    """Show PD controller in action over time."""
    
    print("\n=== REAL-WORLD PD EXAMPLE ===\n")
    print("Robot trying to reach target [3.0, 2.0] from [0.0, 0.0]")
    print()
    
    # Simulation parameters
    dt = 0.1
    Kp, Kd = 4.0, 1.5
    
    # Initial conditions
    pos = np.array([0.0, 0.0])
    vel = np.array([0.0, 0.0])
    target = np.array([3.0, 2.0])
    
    print("Time | Position    | Velocity    | Error       | Action      ")
    print("-" * 65)
    
    for t in range(8):
        time = t * dt
        
        # PD controller
        pos_error = target - pos
        vel_error = np.array([0.0, 0.0]) - vel  # Want to stop at target
        action = Kp * pos_error + Kd * vel_error
        
        # Simulate robot dynamics (simple integration)
        vel += action * dt
        pos += vel * dt
        
        print(f"{time:4.1f} | [{pos[0]:4.1f}, {pos[1]:4.1f}] | "
              f"[{vel[0]:4.1f}, {vel[1]:4.1f}] | "
              f"[{pos_error[0]:4.1f}, {pos_error[1]:4.1f}] | "
              f"[{action[0]:4.1f}, {action[1]:4.1f}]")

def why_pd_in_robotics():
    """Explain why PD controllers are used in robotics."""
    
    print("\n=== WHY PD CONTROLLERS IN ROBOTICS ===\n")
    
    print("1. SIMPLE & RELIABLE")
    print("   - Only 2 parameters to tune (Kp, Kd)")
    print("   - Well-understood behavior")
    print("   - Computationally efficient")
    
    print("\n2. GOOD PERFORMANCE")
    print("   - Proportional term: Corrects position error")
    print("   - Derivative term: Prevents overshoot, adds damping")
    print("   - Smooth, stable motion")
    
    print("\n3. REAL-TIME CAPABLE")
    print("   - Fast computation: just multiplication & addition")
    print("   - No complex optimization or planning")
    print("   - Deterministic execution time")
    
    print("\n4. WORKS WITH HIERARCHICAL CONTROL")
    print("   - High-level planner: Sets targets")
    print("   - PD controller: Converts targets → actions")
    print("   - Safety filter: Ensures constraints")
    
    print("\n5. PHYSICAL INTUITION")
    print("   - Kp: 'Spring' pulling toward target")
    print("   - Kd: 'Damper' resisting fast motion")
    print("   - Easy to understand and debug")

if __name__ == "__main__":
    explain_pd_controller()
    demonstrate_pd_behavior()
    compare_pd_gains()
    pd_vs_other_controllers()
    real_world_pd_example()
    why_pd_in_robotics()