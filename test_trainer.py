
import torch
import numpy as np
from src.environment import WarehouseEnvironment
from src.planning.fsm_planner import FSMState, FSMTransition, FSMAutomaton
from src.core.policy import SubgoalConditionedPolicy
from src.cbf import CBFNetwork
from src.clf import CLFNetwork
from src.models import EnsembleDynamics, ReplayBuffer
from src.training.integrated_trainer import FSMCBFCLFTrainer

print("Testing IntegratedTrainer setup...")

# Create environment
env = WarehouseEnvironment()
print(f"✓ Environment created")

# Create FSM
def at_intermediate_goal(state):
    intermediate = np.array([5.0, 5.0])
    return np.linalg.norm(state[:2] - intermediate) < 1.0

def at_final_goal(state):
    goal = np.array([10.5, 8.5])
    return np.linalg.norm(state[:2] - goal) < 0.5

predicates = {
    'at_intermediate': at_intermediate_goal,
    'at_final': at_final_goal
}

fsm_states = [
    FSMState(id='start', subgoal=np.array([5.0, 5.0, 0.0, 0.0]), is_goal=False),
    FSMState(id='intermediate', subgoal=np.array([10.5, 8.5, 0.0, 0.0]), is_goal=False),
    FSMState(id='goal', subgoal=np.array([10.5, 8.5, 0.0, 0.0]), is_goal=True)
]

fsm_transitions = [
    FSMTransition('start', 'intermediate', 'at_intermediate'),
    FSMTransition('intermediate', 'goal', 'at_final')
]

fsm = FSMAutomaton(fsm_states, fsm_transitions, 'start', predicates)
print(f"✓ FSM created with {len(fsm_states)} states")

# Create components
policy = SubgoalConditionedPolicy(state_dim=4, action_dim=2, subgoal_dim=4, hidden_dims=(128, 128), device='cpu')
cbf = CBFNetwork(state_dim=4, hidden_dims=(64, 64), device='cpu')
clf = CLFNetwork(state_dim=4, hidden_dims=(64, 64), device='cpu')
dynamics = EnsembleDynamics(num_models=3, state_dim=4, action_dim=2, hidden_dims=(128, 128), device='cpu')
buffer = ReplayBuffer(capacity=10000, state_dim=4, action_dim=2)
print(f"✓ Components created")

# Create optimizers
policy_opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
cbf_opt = torch.optim.Adam(cbf.parameters(), lr=1e-3)
clf_opt = torch.optim.Adam(clf.parameters(), lr=1e-3)
dynamics_opt = torch.optim.Adam(dynamics.parameters(), lr=1e-3)

config = {
    'batch_size': 32,
    'model_update_freq': 5,
    'cbf_update_freq': 10,
    'clf_update_freq': 10,
    'lambda_cbf': 1.0,
    'lambda_clf': 1.0,
    'epsilon': 0.1
}

# Create trainer
trainer = FSMCBFCLFTrainer(
    fsm, policy, cbf, clf, dynamics, buffer,
    policy_opt, cbf_opt, clf_opt, dynamics_opt,
    config, device='cpu'
)
print(f"✓ Trainer created")

# Run 1 quick episode
print("
Running test episode...")
stats = trainer.training_episode(env, max_steps=20)
print(f"Episode result: {stats}")

print("
✅ IntegratedTrainer test successful!")
