"""
Detailed dry run of the safe planning pipeline with actual values.
"""

import torch
import numpy as np
from src import create_trainer, Transition

def detailed_dry_run():
    """Step-by-step dry run with actual numerical values."""
    
    print("=== LONG-HORIZON SAFE PLANNING - DETAILED DRY RUN ===\n")
    
    # ===== STEP 1: INITIALIZATION =====
    print("STEP 1: INITIALIZATION")
    print("-" * 50)
    
    state_dim = 4  # [x, y, vx, vy]
    action_dim = 2  # [ax, ay]
    
    trainer = create_trainer(
        state_dim=state_dim,
        action_dim=action_dim,
        device="cpu",
        batch_size=4,  # Small batch for demo
        cbf_update_freq=5,
        clf_update_freq=5
    )
    
    print(f"✓ Created trainer: state_dim={state_dim}, action_dim={action_dim}")
    print(f"✓ CBF ensemble: {trainer.cbf_ensemble.num_models} models")
    print(f"✓ CLF ensemble: {trainer.clf_ensemble.num_models} models") 
    print(f"✓ Dynamics ensemble: {trainer.dynamics_ensemble.num_models} models")
    print(f"✓ Replay buffer capacity: {trainer.replay_buffer.capacity}")
    print()
    
    # ===== STEP 2: FIRST TRANSITION =====
    print("STEP 2: FIRST TRANSITION")
    print("-" * 50)
    
    # Initial state: robot at position (1.0, 0.5) with velocity (0.1, -0.2)
    state = np.array([1.0, 0.5, 0.1, -0.2])
    action = np.array([0.05, 0.1])  # Small acceleration
    next_state = np.array([1.05, 0.48, 0.15, -0.1])  # After dynamics
    
    print(f"Input state: {state}")
    print(f"Input action: {action}")
    print(f"Next state: {next_state}")
    
    # Create transition
    transition = Transition(
        state=state,
        action=action,
        next_state=next_state,
        reward=-0.5,  # Distance-based reward
        done=False,
        is_safe=True,   # No collision
        is_goal=False   # Not at goal yet
    )
    
    print(f"Transition created:")
    print(f"  - is_safe: {transition.is_safe}")
    print(f"  - is_goal: {transition.is_goal}")
    print(f"  - reward: {transition.reward}")
    
    # Add to trainer
    print(f"\nBefore adding transition:")
    print(f"  - step_count: {trainer.step_count}")
    print(f"  - buffer_size: {len(trainer.replay_buffer)}")
    print(f"  - safe_states: {len(trainer.labeled_data['safe_states'])}")
    
    trainer.add_transition(transition)
    
    print(f"\nAfter adding transition:")
    print(f"  - step_count: {trainer.step_count}")
    print(f"  - buffer_size: {len(trainer.replay_buffer)}")
    print(f"  - safe_states: {len(trainer.labeled_data['safe_states'])}")
    print(f"  - transitions: {len(trainer.labeled_data['transitions'])}")
    print()
    
    # ===== STEP 3: NETWORK PREDICTIONS (UNTRAINED) =====
    print("STEP 3: INITIAL NETWORK PREDICTIONS (UNTRAINED)")
    print("-" * 50)
    
    state_tensor = torch.FloatTensor(state).unsqueeze(0)  # [1, 4]
    action_tensor = torch.FloatTensor(action).unsqueeze(0)  # [1, 2]
    
    # CBF predictions
    with torch.no_grad():
        cbf_values = trainer.cbf_ensemble(state_tensor)
        cbf_all = trainer.cbf_ensemble.forward_all(state_tensor)
        cbf_uncertainty = trainer.cbf_ensemble.uncertainty(state_tensor)
    
    print(f"CBF predictions for state {state}:")
    print(f"  - Mean CBF value: {cbf_values.item():.4f}")
    print(f"  - Individual models: {cbf_all.squeeze().tolist()}")
    print(f"  - Uncertainty (std): {cbf_uncertainty.item():.4f}")
    
    # CLF predictions
    with torch.no_grad():
        clf_values = trainer.clf_ensemble(state_tensor)
        clf_all = trainer.clf_ensemble.forward_all(state_tensor)
        clf_uncertainty = trainer.clf_ensemble.uncertainty(state_tensor)
    
    print(f"\nCLF predictions for state {state}:")
    print(f"  - Mean CLF value: {clf_values.item():.4f}")
    print(f"  - Individual models: {clf_all.squeeze().tolist()}")
    print(f"  - Uncertainty (std): {clf_uncertainty.item():.4f}")
    
    # Dynamics predictions
    with torch.no_grad():
        dynamics_pred = trainer.dynamics_ensemble(state_tensor, action_tensor)
        dynamics_all = trainer.dynamics_ensemble.forward_all(state_tensor, action_tensor)
        dynamics_uncertainty = trainer.dynamics_ensemble.uncertainty(state_tensor, action_tensor)
    
    print(f"\nDynamics predictions for state {state}, action {action}:")
    print(f"  - Mean prediction: {dynamics_pred.squeeze().tolist()}")
    print(f"  - Actual next state: {next_state.tolist()}")
    print(f"  - Prediction error: {(dynamics_pred.squeeze() - torch.FloatTensor(next_state)).abs().mean().item():.4f}")
    print(f"  - Uncertainty (std): {dynamics_uncertainty.mean().item():.4f}")
    print()
    
    # ===== STEP 4: ADD MORE TRANSITIONS =====
    print("STEP 4: ADDING MORE TRANSITIONS")
    print("-" * 50)
    
    transitions_data = [
        # Safe transitions
        ([1.05, 0.48, 0.15, -0.1], [0.02, 0.08], [1.07, 0.46, 0.17, -0.02], True, False),
        ([1.07, 0.46, 0.17, -0.02], [-0.01, 0.05], [1.06, 0.44, 0.16, 0.03], True, False),
        ([1.06, 0.44, 0.16, 0.03], [-0.05, -0.02], [1.01, 0.42, 0.11, 0.01], True, False),
        
        # Unsafe transition (collision)
        ([2.8, 1.9, 0.3, 0.2], [0.1, 0.1], [2.9, 2.0, 0.4, 0.3], False, False),
        
        # Goal transition
        ([0.1, 0.05, -0.02, -0.01], [-0.01, -0.005], [0.09, 0.045, -0.03, -0.015], True, True),
    ]
    
    for i, (s, a, ns, is_safe, is_goal) in enumerate(transitions_data):
        transition = Transition(
            state=np.array(s),
            action=np.array(a),
            next_state=np.array(ns),
            reward=-np.linalg.norm(ns[:2]),  # Distance to origin
            done=is_goal,
            is_safe=is_safe,
            is_goal=is_goal
        )
        trainer.add_transition(transition)
        
        print(f"Added transition {i+1}: safe={is_safe}, goal={is_goal}")
    
    print(f"\nAfter adding {len(transitions_data)} more transitions:")
    print(f"  - Total steps: {trainer.step_count}")
    print(f"  - Buffer size: {len(trainer.replay_buffer)}")
    print(f"  - Safe states: {len(trainer.labeled_data['safe_states'])}")
    print(f"  - Unsafe states: {len(trainer.labeled_data['unsafe_states'])}")
    print(f"  - Goal states: {len(trainer.labeled_data['goal_states'])}")
    print()
    
    # ===== STEP 5: TRAINING UPDATES =====
    print("STEP 5: TRAINING UPDATES")
    print("-" * 50)
    
    # Force CBF update
    if len(trainer.labeled_data['safe_states']) > 0 and len(trainer.labeled_data['unsafe_states']) > 0:
        print("Triggering CBF update...")
        
        # Get training data
        safe_states = torch.FloatTensor(trainer.labeled_data['safe_states'])
        unsafe_states = torch.FloatTensor(trainer.labeled_data['unsafe_states'])
        
        # Sample transitions for constraint learning
        if len(trainer.replay_buffer) >= 4:
            batch = trainer.replay_buffer.sample(4)
            states = batch['states']
            next_states = batch['next_states']
            
            print(f"CBF training data:")
            print(f"  - Safe states: {safe_states.shape}")
            print(f"  - Unsafe states: {unsafe_states.shape}")
            print(f"  - Transition states: {states.shape}")
            
            # Train one CBF model
            cbf_trainer = trainer.cbf_trainers[0]
            losses = cbf_trainer.train_step(safe_states, unsafe_states, states, next_states)
            
            print(f"CBF losses after training:")
            for loss_name, loss_value in losses.items():
                print(f"  - {loss_name}: {loss_value:.6f}")
    
    # Force CLF update
    if len(trainer.labeled_data['goal_states']) > 0:
        print(f"\nTriggering CLF update...")
        
        goal_states = torch.FloatTensor(trainer.labeled_data['goal_states'])
        
        if len(trainer.replay_buffer) >= 4:
            batch = trainer.replay_buffer.sample(4)
            states = batch['states']
            next_states = batch['next_states']
            
            print(f"CLF training data:")
            print(f"  - Goal states: {goal_states.shape}")
            print(f"  - Transition states: {states.shape}")
            
            # Train one CLF model
            clf_trainer = trainer.clf_trainers[0]
            losses = clf_trainer.train_step(goal_states, states, next_states)
            
            print(f"CLF losses after training:")
            for loss_name, loss_value in losses.items():
                print(f"  - {loss_name}: {loss_value:.6f}")
    
    # Dynamics update
    if len(trainer.replay_buffer) >= 4:
        print(f"\nTriggering dynamics update...")
        losses = trainer.model_learner.update_model()
        print(f"Dynamics losses after training:")
        print(f"  - Total model loss: {losses['model_loss']:.6f}")
        print(f"  - Individual losses: {[f'{l:.6f}' for l in losses['individual_losses']]}")
    
    print()
    
    # ===== STEP 6: NETWORK PREDICTIONS (AFTER TRAINING) =====
    print("STEP 6: NETWORK PREDICTIONS AFTER TRAINING")
    print("-" * 50)
    
    test_state = np.array([0.5, 0.3, 0.05, -0.05])
    test_action = np.array([0.02, 0.03])
    
    state_tensor = torch.FloatTensor(test_state).unsqueeze(0)
    action_tensor = torch.FloatTensor(test_action).unsqueeze(0)
    
    with torch.no_grad():
        # CBF after training
        cbf_values_trained = trainer.cbf_ensemble(state_tensor)
        print(f"CBF value for test state {test_state}: {cbf_values_trained.item():.4f}")
        
        # CLF after training
        clf_values_trained = trainer.clf_ensemble(state_tensor)
        print(f"CLF value for test state {test_state}: {clf_values_trained.item():.4f}")
        
        # Dynamics after training
        dynamics_pred_trained = trainer.dynamics_ensemble(state_tensor, action_tensor)
        print(f"Dynamics prediction: {dynamics_pred_trained.squeeze().tolist()}")
        
        # Check constraints
        next_state_tensor = dynamics_pred_trained
        cbf_constraint = trainer.cbf_ensemble.cbf_constraint(state_tensor, next_state_tensor)
        clf_constraint = trainer.clf_ensemble.clf_constraint(state_tensor, next_state_tensor)
        
        print(f"CBF constraint violation: {cbf_constraint.item():.6f}")
        print(f"CLF constraint violation: {clf_constraint.item():.6f}")
    
    print()
    
    # ===== STEP 7: SAFE ACTION FILTERING =====
    print("STEP 7: SAFE ACTION FILTERING")
    print("-" * 50)
    
    test_state = np.array([1.5, 1.0, 0.2, 0.1])
    proposed_action = np.array([0.1, 0.05])
    
    print(f"Test state: {test_state}")
    print(f"Proposed action: {proposed_action}")
    
    # Get safe action
    safe_action = trainer.get_safe_action(test_state, proposed_action)
    
    print(f"Safe action: {safe_action.numpy()}")
    print(f"Action modification: {(safe_action.numpy() - proposed_action)}")
    
    # Get safety metrics
    state_batch = torch.FloatTensor(test_state).unsqueeze(0)
    action_batch = torch.FloatTensor(proposed_action).unsqueeze(0)
    
    metrics = trainer.controller.get_safety_feasibility_metrics(
        state_batch, action_batch, trainer.dynamics_ensemble
    )
    
    print(f"\nSafety metrics:")
    print(f"  - CBF value: {metrics['cbf_values'].item():.4f}")
    print(f"  - CLF value: {metrics['clf_values'].item():.4f}")
    print(f"  - Is safe: {metrics['is_safe'].item()}")
    print(f"  - Is near goal: {metrics['is_near_goal'].item()}")
    print(f"  - CBF constraint: {metrics['cbf_constraints'].item():.6f}")
    print(f"  - CLF constraint: {metrics['clf_constraints'].item():.6f}")
    
    print()
    
    # ===== STEP 8: TRAINING SUMMARY =====
    print("STEP 8: TRAINING SUMMARY")
    print("-" * 50)
    
    summary = trainer.get_training_summary()
    
    print(f"Training Summary:")
    print(f"  - Total steps: {summary.step_count}")
    print(f"  - Buffer size: {summary.buffer_size}")
    print(f"  - Avg model uncertainty: {summary.avg_model_uncertainty:.6f}")
    
    # Evaluate on batch
    if len(trainer.replay_buffer) >= 8:
        batch = trainer.replay_buffer.sample(8)
        states = batch['states']
        actions = batch['actions']
        
        safety_metrics = trainer.evaluate(states, actions)
        
        print(f"\nEvaluation Metrics:")
        print(f"  - Safety rate: {safety_metrics.safety_rate:.3f}")
        print(f"  - Goal proximity rate: {safety_metrics.goal_proximity_rate:.3f}")
        print(f"  - Avg CBF value: {safety_metrics.avg_cbf_value:.4f}")
        print(f"  - Avg CLF value: {safety_metrics.avg_clf_value:.4f}")
        print(f"  - CBF violations: {safety_metrics.cbf_constraint_violations:.6f}")
        print(f"  - CLF violations: {safety_metrics.clf_constraint_violations:.6f}")
    
    print("\n=== DRY RUN COMPLETE ===")

if __name__ == "__main__":
    detailed_dry_run()