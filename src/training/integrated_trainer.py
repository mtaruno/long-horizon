"""
Integrated training loop (Algorithm 2) with FSM, Policy, CBF, CLF, Dynamics.
"""

import torch
import numpy as np
from typing import Dict, Optional
import logging

from ..core.policy import SubgoalConditionedPolicy, PolicyTrainer
from ..planning.fsm_planner import FSMAutomaton, FSMPruner


class FSMCBFCLFTrainer:
    """
    Algorithm 2: Joint Model-Policy Training with Certificate Critics
    
    Co-trains:
    - Policy π_θ(s, g)
    - Dynamics P̂_θ(s, a)
    - CBF h_φ(s)
    - CLF V_ψ(s)
    """
    
    def __init__(
        self,
        fsm: FSMAutomaton,
        policy: SubgoalConditionedPolicy,
        cbf: torch.nn.Module,
        clf: torch.nn.Module,
        dynamics: torch.nn.Module,
        replay_buffer,
        policy_optimizer: torch.optim.Optimizer,
        cbf_optimizer: torch.optim.Optimizer,
        clf_optimizer: torch.optim.Optimizer,
        dynamics_optimizer: torch.optim.Optimizer,
        config: dict,
        device: str = "cpu"
    ):
        self.fsm = fsm
        self.policy = policy
        self.cbf = cbf
        self.clf = clf
        self.dynamics = dynamics
        self.buffer = replay_buffer
        
        self.policy_trainer = PolicyTrainer(
            policy, dynamics, cbf, clf, policy_optimizer,
            config.get("lambda_cbf", 1.0),
            config.get("lambda_clf", 1.0),
            config.get("epsilon", 0.1),
            device
        )
        
        self.cbf_opt = cbf_optimizer
        self.clf_opt = clf_optimizer
        self.dyn_opt = dynamics_optimizer
        
        self.config = config
        self.device = device
        self.step_count = 0
        
        # Data storage
        self.safe_states = []
        self.unsafe_states = []
        self.goal_states = []
        
        self.logger = logging.getLogger(__name__)
    
    def training_episode(self, env, max_steps: int = 1000) -> Dict:
        """
        Execute one training episode following Algorithm 2.
        Args:
            env: Environment with reset(), step(action) interface
            max_steps: Maximum steps per episode
            
        Returns:
            Episode statistics
        """
        state = env.reset()
        self.fsm.current_state_id = list(self.fsm.states.keys())[0]  # Reset FSM
        
        episode_reward = 0
        episode_steps = 0
        
        for k in range(max_steps):
            # Get current subgoal from FSM
            subgoal = self.fsm.get_current_subgoal()
            
            # Policy proposes action: a_k = π_θ(s_k, g_k)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            subgoal_tensor = torch.FloatTensor(subgoal).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                action = self.policy(state_tensor, subgoal_tensor).squeeze(0).cpu().numpy()
                # Exploration: 10% random actions
                if np.random.rand() < 0.1:
                    action = np.random.uniform(-0.5, 0.5, len(action))
            
            # Execute action
            next_state, reward, done, info = env.step(action)
            
            # Label safety and goal
            is_safe = not info.get("collision", False)
            is_goal = info.get("success", False)
            
            # Store transition
            self.buffer.push(state, action, next_state, reward, done)
            
            if is_safe:
                self.safe_states.append(state)
            else:
                self.unsafe_states.append(state)
            
            if is_goal:
                self.goal_states.append(next_state)
            
            # Periodic updates
            if self.step_count % self.config.get("model_update_freq", 5) == 0:
                self._update_dynamics()
            
            if self.step_count % self.config.get("cbf_update_freq", 10) == 0:
                self._update_cbf()
            
            if self.step_count % self.config.get("clf_update_freq", 10) == 0:
                self._update_clf()
            
            # Policy update
            self._update_policy()
            
            # FSM transition
            transitioned = self.fsm.step(next_state)
            if transitioned:
                self.logger.info(f"FSM transitioned to {self.fsm.current_state_id}")
            
            # Check goal
            if self.fsm.is_goal_reached():
                self.logger.info(f"Goal reached at step {k}")
                break
            
            state = next_state
            episode_reward += reward
            episode_steps += 1
            self.step_count += 1
            
            if done:
                break
        
        return {
            "reward": episode_reward,
            "steps": episode_steps,
            "goal_reached": self.fsm.is_goal_reached()
        }
    
    def _update_dynamics(self):
        """Update dynamics model: L_dyn = ||P̂_θ(s,a) - s'||²"""
        if len(self.buffer) < self.config.get("batch_size", 64):
            return
        
        batch = self.buffer.sample(self.config["batch_size"])
        states = batch["states"].to(self.device)
        actions = batch["actions"].to(self.device)
        next_states = batch["next_states"].to(self.device)
        
        self.dyn_opt.zero_grad()
        pred_next = self.dynamics(states, actions)
        loss = torch.mean((pred_next - next_states) ** 2)
        loss.backward()
        self.dyn_opt.step()
    
    def _update_cbf(self):
        """Update CBF: L_CBF with safe/unsafe/constraint terms"""
        if len(self.safe_states) < 10 or len(self.unsafe_states) < 10:
            return
        
        safe = torch.FloatTensor(self.safe_states[-100:]).to(self.device)
        unsafe = torch.FloatTensor(self.unsafe_states[-100:]).to(self.device)
        
        batch = self.buffer.sample(min(len(self.buffer), 64))
        states = batch["states"].to(self.device)
        next_states = batch["next_states"].to(self.device)
        
        self.cbf_opt.zero_grad()
        
        # Safe loss: h(s) >= 0
        h_safe = self.cbf(safe).squeeze(-1)
        loss_safe = torch.mean(torch.clamp(-h_safe, min=0.0) ** 2)
        
        # Unsafe loss: h(s) < 0
        h_unsafe = self.cbf(unsafe).squeeze(-1)
        loss_unsafe = torch.mean(torch.clamp(h_unsafe, min=0.0) ** 2)
        
        # Constraint loss: h(s') - h(s) >= -α·h(s)
        h_curr = self.cbf(states).squeeze(-1)
        h_next = self.cbf(next_states).squeeze(-1)
        alpha = 0.1
        constraint_viol = torch.clamp(-alpha * h_curr - (h_next - h_curr), min=0.0)
        loss_constraint = torch.mean(constraint_viol ** 2)
        
        loss = loss_safe + loss_unsafe + loss_constraint
        loss.backward()
        self.cbf_opt.step()
    
    def _update_clf(self):
        """Update CLF: L_CLF with goal/constraint terms"""
        if len(self.goal_states) < 5:
            return
        
        goals = torch.FloatTensor(self.goal_states[-50:]).to(self.device)
        
        batch = self.buffer.sample(min(len(self.buffer), 64))
        states = batch["states"].to(self.device)
        next_states = batch["next_states"].to(self.device)
        
        self.clf_opt.zero_grad()
        
        # Goal loss: V(g) = 0
        V_goal = self.clf(goals).squeeze(-1)
        loss_goal = torch.mean(V_goal ** 2)
        
        # Constraint loss: V(s') - V(s) <= -β·V(s) + δ
        V_curr = self.clf(states).squeeze(-1)
        V_next = self.clf(next_states).squeeze(-1)
        beta, delta = 0.1, 0.01
        constraint_viol = torch.clamp(V_next - V_curr + beta * V_curr - delta, min=0.0)
        loss_constraint = torch.mean(constraint_viol ** 2)
        
        # Positive loss: V(s) > 0
        loss_positive = torch.mean(torch.clamp(0.01 - V_curr, min=0.0) ** 2)
        
        loss = loss_goal + loss_constraint + 0.1 * loss_positive
        loss.backward()
        self.clf_opt.step()
    
    def _update_policy(self):
        """Update policy: L_actor = ℓ(ŝ', g) + λ_cbf·c_cbf + λ_clf·c_clf + reward (MODEL-BASED)"""
        if len(self.buffer) < self.config.get("batch_size", 64):
            return

        batch = self.buffer.sample(self.config["batch_size"])
        states = batch["states"].to(self.device)
        rewards = batch["rewards"].to(self.device)

        # Get subgoals (simplified: use current FSM subgoal for all)
        subgoal = self.fsm.get_current_subgoal()
        subgoals = torch.FloatTensor(
            np.tile(subgoal, (states.shape[0], 1))
        ).to(self.device)

        # Model-based update using dynamics predictions
        self.policy_trainer.train_step(states, subgoals, rewards, next_states=None, use_model_free=False)
    
    def prune_fsm(self) -> FSMAutomaton:
        """Prune FSM using learned certificates (Algorithm 1)"""
        pruner = FSMPruner(
            self.cbf, self.clf, self.dynamics,
            epsilon_safe=0.0,
            epsilon_feas=0.1,
            device=self.device
        )
        return pruner.prune(self.fsm)
