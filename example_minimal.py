"""
Minimal example demonstrating FSM-CBF-CLF framework.
"""

import numpy as np
import torch
import torch.optim as optim

# Import existing components
from src.cbf import EnsembleCBF
from src.clf import EnsembleCLF
from src.models import EnsembleDynamics, ReplayBuffer

# Import new modular components
from src.core.policy import SubgoalConditionedPolicy
from src.planning.fsm_planner import create_simple_navigation_fsm
from src.training.integrated_trainer import FSMCBFCLFTrainer


class SimpleWarehouseEnv:
    """Minimal warehouse environment for testing"""
    
    def __init__(self):
        self.state_dim = 4  # [x, y, vx, vy]
        self.action_dim = 2  # [ax, ay]
        self.dt = 0.1
        self.goal = np.array([10.0, 8.0])
        self.obstacles = [np.array([5.0, 5.0])]
        self.obstacle_radius = 0.5
        
    def reset(self):
        self.state = np.array([0.0, 0.0, 0.0, 0.0])
        return self.state.copy()
    
    def step(self, action):
        # Clip action
        action = np.clip(action, -2.0, 2.0)
        
        # Update state: simple integrator dynamics
        pos = self.state[:2]
        vel = self.state[2:]
        
        vel_new = vel + action * self.dt
        vel_new = np.clip(vel_new, -1.0, 1.0)
        
        pos_new = pos + vel_new * self.dt
        
        self.state = np.concatenate([pos_new, vel_new])
        
        # Check collision
        collision = False
        for obs in self.obstacles:
            if np.linalg.norm(pos_new - obs) < self.obstacle_radius:
                collision = True
        
        # Check success
        success = np.linalg.norm(pos_new - self.goal) < 0.2
        
        # Reward
        reward = -np.linalg.norm(pos_new - self.goal)
        if collision:
            reward -= 10.0
        if success:
            reward += 100.0
        
        done = collision or success
        
        info = {"collision": collision, "success": success}
        
        return self.state.copy(), reward, done, info


def main():
    # Configuration
    state_dim = 4
    action_dim = 2
    subgoal_dim = 2  # [x, y] position
    device = "cpu"
    
    print("=" * 60)
    print("Minimal FSM-CBF-CLF Framework Example")
    print("=" * 60)
    
    # 1. Create environment
    env = SimpleWarehouseEnv()
    print(f"\n✓ Environment created (goal: {env.goal})")
    
    # 2. Create FSM
    fsm = create_simple_navigation_fsm(
        start_pos=np.array([0.0, 0.0]),
        goal_pos=env.goal,
        state_dim=state_dim
    )
    print(f"✓ FSM created with {len(fsm.states)} states")
    
    # 3. Create networks
    policy = SubgoalConditionedPolicy(state_dim, action_dim, subgoal_dim, device=device)
    cbf = EnsembleCBF(num_models=3, state_dim=state_dim, device=device)
    clf = EnsembleCLF(num_models=3, state_dim=state_dim, device=device)
    dynamics = EnsembleDynamics(num_models=3, state_dim=state_dim, action_dim=action_dim, device=device)
    print("✓ Neural networks initialized")
    
    # 4. Create optimizers
    policy_opt = optim.Adam(policy.parameters(), lr=1e-3)
    cbf_opt = optim.Adam(cbf.parameters(), lr=1e-3)
    clf_opt = optim.Adam(clf.parameters(), lr=1e-3)
    dyn_opt = optim.Adam(dynamics.parameters(), lr=1e-3)
    print("✓ Optimizers created")
    
    # 5. Create replay buffer
    buffer = ReplayBuffer(capacity=10000, state_dim=state_dim, action_dim=action_dim)
    print("✓ Replay buffer initialized")
    
    # 6. Create integrated trainer
    config = {
        "lambda_cbf": 1.0,
        "lambda_clf": 1.0,
        "epsilon": 0.1,
        "batch_size": 64,
        "model_update_freq": 5,
        "cbf_update_freq": 10,
        "clf_update_freq": 10
    }
    
    trainer = FSMCBFCLFTrainer(
        fsm=fsm,
        policy=policy,
        cbf=cbf,
        clf=clf,
        dynamics=dynamics,
        replay_buffer=buffer,
        policy_optimizer=policy_opt,
        cbf_optimizer=cbf_opt,
        clf_optimizer=clf_opt,
        dynamics_optimizer=dyn_opt,
        config=config,
        device=device
    )
    print("✓ Integrated trainer created")
    
    # 7. Training loop
    print("\n" + "=" * 60)
    print("Starting Training (Algorithm 2)")
    print("=" * 60)
    
    num_episodes = 5
    for episode in range(num_episodes):
        stats = trainer.training_episode(env, max_steps=200)
        
        print(f"\nEpisode {episode + 1}/{num_episodes}:")
        print(f"  Reward: {stats['reward']:.2f}")
        print(f"  Steps: {stats['steps']}")
        print(f"  Goal Reached: {stats['goal_reached']}")
        print(f"  Buffer Size: {len(buffer)}")
    
    # 8. FSM Pruning (Algorithm 1)
    print("\n" + "=" * 60)
    print("FSM Pruning (Algorithm 1)")
    print("=" * 60)
    
    if len(buffer) > 100:
        pruned_fsm = trainer.prune_fsm()
        print(f"Original FSM: {len(fsm.states)} states, {len(fsm.transitions)} transitions")
        print(f"Pruned FSM: {len(pruned_fsm.states)} states, {len(pruned_fsm.transitions)} transitions")
    else:
        print("Not enough data for pruning (need >100 samples)")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
