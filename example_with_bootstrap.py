"""
Example showing the complete training process with bootstrapping phase.
"""

import numpy as np
import torch
import torch.optim as optim

from src.cbf import EnsembleCBF
from src.clf import EnsembleCLF
from src.models import EnsembleDynamics, ReplayBuffer
from src.core.policy import SubgoalConditionedPolicy
from src.planning.fsm_planner import FSMState, FSMTransition, FSMAutomaton
from src.training.integrated_trainer import FSMCBFCLFTrainer
from src.dataset import create_warehouse_dataset


def load_dataset_into_buffer(buffer):
    """
    Phase 1: Load pre-generated dataset from dataset module.
    
    Returns:
        safe_states, unsafe_states, goal_states
    """
    print("\n" + "="*60)
    print("PHASE 1: Loading Dataset")
    print("="*60)
    
    transitions = create_warehouse_dataset(num_transitions=1000)
    
    safe_states = []
    unsafe_states = []
    goal_states = []
    
    for t in transitions:
        buffer.push(t.state, t.action, t.next_state, t.reward, t.done)
        
        if t.is_safe:
            safe_states.append(t.state)
        else:
            unsafe_states.append(t.state)
        
        if t.is_goal:
            goal_states.append(t.next_state)
    
    print(f"✓ Loaded {len(buffer)} transitions")
    print(f"✓ Safe states: {len(safe_states)}")
    print(f"✓ Unsafe states: {len(unsafe_states)}")
    print(f"✓ Goal states: {len(goal_states)}")
    
    return safe_states, unsafe_states, goal_states


def pretrain_models(dynamics, cbf, clf, buffer, safe_states, unsafe_states, 
                    goal_states, dyn_opt, cbf_opt, clf_opt, device):
    """
    Phase 2: Pretrain models on collected ground truth data.
    """
    print("\n" + "="*60)
    print("PHASE 2: Pretrain Models on Ground Truth Data")
    print("="*60)
    
    # Pretrain dynamics
    print("\nTraining dynamics model...")
    for epoch in range(50):
        if len(buffer) < 64:
            break
        batch = buffer.sample(min(len(buffer), 64))
        states = batch['states'].to(device)
        actions = batch['actions'].to(device)
        next_states_REAL = batch['next_states'].to(device)  # Ground truth!
        
        dyn_opt.zero_grad()
        pred_next = dynamics(states, actions)
        loss = torch.mean((pred_next - next_states_REAL) ** 2)
        loss.backward()
        dyn_opt.step()
        
        if epoch % 10 == 0:
            print(f"  Epoch {epoch}: Dynamics loss = {loss.item():.4f}")
    
    # Pretrain CBF
    if len(safe_states) > 10 and len(unsafe_states) > 10:
        print("\nTraining CBF...")
        safe = torch.FloatTensor(safe_states).to(device)
        unsafe = torch.FloatTensor(unsafe_states).to(device)
        
        for epoch in range(50):
            cbf_opt.zero_grad()
            
            h_safe = cbf(safe).squeeze(-1)
            h_unsafe = cbf(unsafe).squeeze(-1)
            
            loss_safe = torch.mean(torch.clamp(-h_safe, min=0.0) ** 2)
            loss_unsafe = torch.mean(torch.clamp(h_unsafe, min=0.0) ** 2)
            loss = loss_safe + loss_unsafe
            
            loss.backward()
            cbf_opt.step()
            
            if epoch % 10 == 0:
                print(f"  Epoch {epoch}: CBF loss = {loss.item():.4f}")
    
    # Pretrain CLF
    if len(goal_states) > 5:
        print("\nTraining CLF...")
        goals = torch.FloatTensor(goal_states).to(device)
        
        for epoch in range(50):
            clf_opt.zero_grad()
            
            V_goal = clf(goals).squeeze(-1)
            loss = torch.mean(V_goal ** 2)
            
            loss.backward()
            clf_opt.step()
            
            if epoch % 10 == 0:
                print(f"  Epoch {epoch}: CLF loss = {loss.item():.4f}")
    
    print("\n✓ Pretraining complete")


def evaluate_policy(policy, fsm, workspace_bounds, obstacles, goals, num_episodes=10, max_steps=50, device="cpu"):
    """Evaluate policy success rate"""
    successes = 0
    collisions = 0
    
    for ep in range(num_episodes):
        state = np.array([1.0, 1.0, 0.0, 0.0])  # Start position
        fsm.current_state_id = list(fsm.states.keys())[0]
        
        for step in range(max_steps):
            subgoal = fsm.get_current_subgoal()
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            subgoal_tensor = torch.FloatTensor(subgoal).unsqueeze(0).to(device)
            
            with torch.no_grad():
                action = policy(state_tensor, subgoal_tensor).squeeze(0).cpu().numpy()
                action = np.clip(action * 0.5, -1.0, 1.0)  # Scale down
            
            # Simple dynamics
            vel = state[2:] + action * 0.1
            pos = state[:2] + vel * 0.1
            state = np.concatenate([pos, vel])
            
            # Check collision
            collision = any(np.linalg.norm(pos - obs['center']) < obs['radius'] 
                          for obs in obstacles)
            # Check goal
            success = any(np.linalg.norm(pos - g['center']) < g['radius'] 
                         for g in goals)
            
            if collision:
                collisions += 1
                break
            if success:
                successes += 1
                break
            
            fsm.step(state)
    
    return {
        'success_rate': successes / num_episodes,
        'collision_rate': collisions / num_episodes,
        'completion_rate': (successes + collisions) / num_episodes
    }


def main():
    state_dim = 4
    action_dim = 2
    subgoal_dim = 2
    device = "cpu"
    
    print("="*60)
    print("Complete Training Process with Bootstrapping")
    print("="*60)
    
    # Setup environment parameters matching dataset
    workspace_bounds = (0.0, 10.0, 0.0, 8.0)
    obstacles = [
        {'center': np.array([2.0, 2.0]), 'radius': 0.5},
        {'center': np.array([5.0, 3.0]), 'radius': 0.4},
        {'center': np.array([8.0, 1.5]), 'radius': 0.3},
    ]
    goals = [
        {'center': np.array([9.0, 7.0]), 'radius': 0.3},
        {'center': np.array([1.0, 7.0]), 'radius': 0.3},
    ]
    
    # Simple env for trainer
    class SimpleEnv:
        def __init__(self):
            self.state = np.array([1.0, 1.0, 0.0, 0.0])
            self.action_dim = 2
        
        def reset(self):
            self.state = np.array([1.0, 1.0, 0.0, 0.0])
            return self.state.copy()
        
        def step(self, action):
            action = np.clip(action, -1.0, 1.0)
            vel = self.state[2:] + action * 0.1
            pos = self.state[:2] + vel * 0.1
            self.state = np.concatenate([pos, vel])
            
            collision = any(np.linalg.norm(pos - obs['center']) < obs['radius'] 
                          for obs in obstacles)
            success = any(np.linalg.norm(pos - g['center']) < g['radius'] 
                         for g in goals)
            
            reward = -min(np.linalg.norm(pos - g['center']) for g in goals)
            if collision:
                reward -= 10.0
            if success:
                reward += 50.0
            
            info = {"collision": collision, "success": success}
            return self.state.copy(), reward, collision or success, info
    
    env = SimpleEnv()
    buffer = ReplayBuffer(capacity=10000, state_dim=state_dim, action_dim=action_dim)
    
    policy = SubgoalConditionedPolicy(state_dim, action_dim, subgoal_dim, device=device)
    cbf = EnsembleCBF(num_models=3, state_dim=state_dim, device=device)
    clf = EnsembleCLF(num_models=3, state_dim=state_dim, device=device)
    dynamics = EnsembleDynamics(num_models=3, state_dim=state_dim, 
                                action_dim=action_dim, device=device)
    
    policy_opt = optim.Adam(policy.parameters(), lr=3e-3)
    cbf_opt = optim.Adam(cbf.parameters(), lr=1e-3)
    clf_opt = optim.Adam(clf.parameters(), lr=1e-3)
    dyn_opt = optim.Adam(dynamics.parameters(), lr=1e-3)
    
    # PHASE 1: Load dataset
    safe_states, unsafe_states, goal_states = load_dataset_into_buffer(buffer)
    
    # PHASE 2: Pretrain on ground truth data
    pretrain_models(dynamics, cbf, clf, buffer, safe_states, unsafe_states,
                   goal_states, dyn_opt, cbf_opt, clf_opt, device)
    
    # PHASE 3: Online learning with policy
    print("\n" + "="*60)
    print("PHASE 3: Online Learning with Policy")
    print("="*60)
    
    # Create FSM matching dataset goals
    states = [
        FSMState(id="navigate", subgoal=np.array([9.0, 7.0]), is_goal=False),
        FSMState(id="goal", subgoal=np.array([9.0, 7.0]), is_goal=True)
    ]
    transitions = [
        FSMTransition("navigate", "goal", "at_goal")
    ]
    predicates = {
        "at_goal": lambda s: np.linalg.norm(s[:2] - np.array([9.0, 7.0])) < 0.5
    }
    fsm = FSMAutomaton(states, transitions, "navigate", predicates)
    
    config = {
        "lambda_cbf": 0.5,
        "lambda_clf": 2.0,
        "epsilon": 0.1,
        "batch_size": 64,
        "model_update_freq": 5,
        "cbf_update_freq": 10,
        "clf_update_freq": 10
    }
    
    trainer = FSMCBFCLFTrainer(
        fsm=fsm, policy=policy, cbf=cbf, clf=clf, dynamics=dynamics,
        replay_buffer=buffer, policy_optimizer=policy_opt,
        cbf_optimizer=cbf_opt, clf_optimizer=clf_opt,
        dynamics_optimizer=dyn_opt, config=config, device=device
    )
    
    # Evaluate before training
    print("\nEvaluating initial policy...")
    metrics = evaluate_policy(policy, fsm, workspace_bounds, obstacles, goals, num_episodes=10, device=device)
    print(f"Initial Success Rate: {metrics['success_rate']*100:.1f}%")
    print(f"Initial Collision Rate: {metrics['collision_rate']*100:.1f}%")
    
    # Training
    print("\nTraining...")
    for episode in range(100):
        stats = trainer.training_episode(env, max_steps=100)
        if episode % 10 == 0:
            print(f"Episode {episode + 1}: Reward={stats['reward']:.1f}, Goal={stats['goal_reached']}")
    
    # Evaluate after training
    print("\nEvaluating trained policy...")
    metrics = evaluate_policy(policy, fsm, workspace_bounds, obstacles, goals, num_episodes=10, device=device)
    print(f"Final Success Rate: {metrics['success_rate']*100:.1f}%")
    print(f"Final Collision Rate: {metrics['collision_rate']*100:.1f}%")
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
