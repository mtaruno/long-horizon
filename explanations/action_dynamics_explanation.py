"""
Explanation of action representation and dynamics prediction necessity.
"""

import numpy as np
import torch

def explain_action_representation():
    """Demonstrate why actions are 2D scalars."""
    
    print("=== ACTION REPRESENTATION EXPLANATION ===\n")
    
    # Robot state: [x, y, vx, vy]
    current_state = np.array([1.0, 0.5, 0.1, -0.2])
    print(f"Robot state: [x={current_state[0]}, y={current_state[1]}, "
          f"vx={current_state[2]}, vy={current_state[3]}]")
    
    # Action: [ax, ay] - acceleration commands
    action = np.array([0.05, 0.1])
    print(f"Action: [ax={action[0]}, ay={action[1]}] m/s²")
    
    # Simple dynamics: next_state = current_state + dt * [vx, vy, ax, ay]
    dt = 0.1  # Time step
    
    # Physics integration
    next_x = current_state[0] + dt * current_state[2]  # x += vx * dt
    next_y = current_state[1] + dt * current_state[3]  # y += vy * dt
    next_vx = current_state[2] + dt * action[0]        # vx += ax * dt
    next_vy = current_state[3] + dt * action[1]        # vy += ay * dt
    
    next_state = np.array([next_x, next_y, next_vx, next_vy])
    
    print(f"\nAfter dt={dt}s:")
    print(f"Next state: [x={next_state[0]:.3f}, y={next_state[1]:.3f}, "
          f"vx={next_state[2]:.3f}, vy={next_state[3]:.3f}]")
    
    print(f"\nPhysical interpretation:")
    print(f"- Robot moved from ({current_state[0]}, {current_state[1]}) "
          f"to ({next_state[0]:.3f}, {next_state[1]:.3f})")
    print(f"- Velocity changed from ({current_state[2]}, {current_state[3]}) "
          f"to ({next_state[2]}, {next_state[3]})")
    
    return current_state, action, next_state

def explain_dynamics_necessity():
    """Demonstrate why we need dynamics prediction for constraints."""
    
    print("\n=== WHY DYNAMICS PREDICTION IS NEEDED ===\n")
    
    current_state, action, true_next_state = explain_action_representation()
    
    # Convert to tensors
    s_current = torch.FloatTensor(current_state).unsqueeze(0)
    s_next_true = torch.FloatTensor(true_next_state).unsqueeze(0)
    
    print("CONSTRAINT EVALUATION PROBLEM:")
    print("-" * 40)
    
    # Mock CBF and CLF functions
    def mock_cbf(state):
        """Mock CBF: h(s) = 2.0 - ||(x,y)||"""
        pos = state[:, :2]  # [x, y]
        return 2.0 - torch.norm(pos, dim=1, keepdim=True)
    
    def mock_clf(state):
        """Mock CLF: V(s) = ||(x,y)||²"""
        pos = state[:, :2]  # [x, y]
        return torch.sum(pos**2, dim=1, keepdim=True)
    
    # Evaluate constraints
    h_current = mock_cbf(s_current)
    h_next = mock_cbf(s_next_true)
    
    V_current = mock_clf(s_current)
    V_next = mock_clf(s_next_true)
    
    print(f"CBF values: h(s_current)={h_current.item():.3f}, h(s_next)={h_next.item():.3f}")
    print(f"CLF values: V(s_current)={V_current.item():.3f}, V(s_next)={V_next.item():.3f}")
    
    # Check constraints
    alpha = 0.1  # CBF parameter
    beta = 0.1   # CLF parameter
    delta = 0.01 # CLF tolerance
    
    cbf_constraint = h_next - h_current + alpha * h_current
    clf_constraint = V_next - V_current + beta * V_current - delta
    
    print(f"\nConstraint checks:")
    print(f"CBF: h_next - h_current >= -α*h_current")
    print(f"     {h_next.item():.3f} - {h_current.item():.3f} >= -{alpha}*{h_current.item():.3f}")
    print(f"     {cbf_constraint.item():.3f} >= 0 → {'✓ SAFE' if cbf_constraint >= 0 else '✗ UNSAFE'}")
    
    print(f"CLF: V_next - V_current <= -β*V_current + δ")
    print(f"     {V_next.item():.3f} - {V_current.item():.3f} <= -{beta}*{V_current.item():.3f} + {delta}")
    print(f"     {clf_constraint.item():.3f} <= 0 → {'✓ FEASIBLE' if clf_constraint <= 0 else '✗ INFEASIBLE'}")
    
    print(f"\nTHE PROBLEM:")
    print(f"- We have: current_state = {current_state}")
    print(f"- We have: proposed_action = {action}")
    print(f"- We need: next_state = ? (to check constraints)")
    print(f"- Solution: Learn dynamics model f(s,a) → s'")
    
    return s_current, action, s_next_true

def demonstrate_dynamics_learning():
    """Show how dynamics model learns the transition function."""
    
    print("\n=== DYNAMICS MODEL LEARNING ===\n")
    
    # Simple linear dynamics model
    class SimpleDynamics(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # Learn: next_state = A * [state; action] + b
            self.linear = torch.nn.Linear(6, 4)  # [4 state + 2 action] → 4 state
            
        def forward(self, state, action):
            input_vec = torch.cat([state, action], dim=1)
            return self.linear(input_vec)
    
    model = SimpleDynamics()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Training data: multiple state-action-next_state examples
    training_data = [
        ([1.0, 0.5, 0.1, -0.2], [0.05, 0.1], [1.01, 0.48, 0.15, -0.1]),
        ([1.01, 0.48, 0.15, -0.1], [0.02, 0.08], [1.025, 0.464, 0.17, -0.02]),
        ([0.5, 0.3, 0.05, -0.05], [0.0, 0.0], [0.505, 0.295, 0.05, -0.05]),
    ]
    
    print("Training dynamics model...")
    for epoch in range(100):
        total_loss = 0
        for state, action, next_state in training_data:
            s = torch.FloatTensor(state).unsqueeze(0)
            a = torch.FloatTensor(action).unsqueeze(0)
            s_next_true = torch.FloatTensor(next_state).unsqueeze(0)
            
            s_next_pred = model(s, a)
            loss = torch.nn.functional.mse_loss(s_next_pred, s_next_true)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Loss = {total_loss:.6f}")
    
    # Test prediction
    test_state = torch.FloatTensor([0.8, 0.4, 0.12, -0.08]).unsqueeze(0)
    test_action = torch.FloatTensor([0.03, 0.05]).unsqueeze(0)
    
    predicted_next = model(test_state, test_action)
    
    print(f"\nTest prediction:")
    print(f"State: {test_state.squeeze().tolist()}")
    print(f"Action: {test_action.squeeze().tolist()}")
    print(f"Predicted next: {predicted_next.squeeze().tolist()}")
    
    print(f"\nNow we can check constraints for ANY proposed action!")
    print(f"1. Get current state")
    print(f"2. Propose action")
    print(f"3. Predict next state using learned model")
    print(f"4. Check CBF/CLF constraints")
    print(f"5. Modify action if constraints violated")

if __name__ == "__main__":
    explain_action_representation()
    explain_dynamics_necessity()
    demonstrate_dynamics_learning()