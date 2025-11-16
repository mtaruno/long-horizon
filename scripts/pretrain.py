import yaml
import numpy as np
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.utils.buffer import ReplayBuffer
from src.core.critics import CBFNetwork, CLFNetwork
from src.utils.visualization import plot_critic_landscapes
from src.environment import WarehouseEnv # For visualization
from src.utils.seeding import set_seed

def main():
    set_seed(42)  # For reproducibility
    # Load config
    with open("config/warehouse_v1.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    pretrain_config = config['pretrain']
    nn_config = config['nn']
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data from buffer
    print(f"Loading data from {config['data_gen']['data_path']}...")
    # Initialize with correct dims from config
    buffer = ReplayBuffer(
        state_dim=config['nn']['state_dim'],
        action_dim=config['nn']['action_dim'],
        subgoal_dim=config['nn']['subgoal_dim'],
        max_size=config['data_gen']['num_transitions'] # Use the size of the data
    )
    buffer.load(config['data_gen']['data_path'])
    
    # Create PyTorch DataLoader
    dataset = TensorDataset(
        torch.from_numpy(buffer.states).float(),
        torch.from_numpy(buffer.subgoals).float(),
        torch.from_numpy(buffer.h_stars).float(),
        torch.from_numpy(buffer.v_stars).float()
    )
    loader = DataLoader(dataset, batch_size=pretrain_config['batch_size'], shuffle=True)
    
    print(f"--- Starting Critic Pre-Training (Phase 1) ---")

    # 1. Initialize Networks
    cbf_net = CBFNetwork(
        state_dim=nn_config['state_dim'],
        hidden_dims=nn_config['hidden_dims']
    ).to(device)
    
    clf_net = CLFNetwork(
        state_dim=nn_config['state_dim'],
        subgoal_dim=nn_config['subgoal_dim'],
        hidden_dims=nn_config['hidden_dims']
    ).to(device)
    
    # 2. Setup Optimizers
    cbf_optim = optim.Adam(cbf_net.parameters(), lr=pretrain_config['lr'])
    clf_optim = optim.Adam(clf_net.parameters(), lr=pretrain_config['lr'])
    
    # 3. Training Loop
    for epoch in range(pretrain_config['epochs']):
        cbf_losses = []
        clf_losses = []
        
        for s, g, h_star, v_star in loader:
            s, g, h_star, v_star = s.to(device), g.to(device), h_star.to(device), v_star.to(device)
            
            # --- Update CBF ---
            cbf_optim.zero_grad()
            cbf_loss = cbf_net.compute_loss_pretrain(s, h_star)
            cbf_loss.backward()
            cbf_optim.step()
            cbf_losses.append(cbf_loss.item())
            
            # --- Update CLF ---
            clf_optim.zero_grad()
            clf_loss = clf_net.compute_loss_pretrain(s, g, v_star)
            clf_loss.backward()
            clf_optim.step()
            clf_losses.append(clf_loss.item())
        
        print(f"Epoch {epoch+1}/{pretrain_config['epochs']} | "
              f"CBF Loss (MSE): {np.mean(cbf_losses):.6f} | "
              f"CLF Loss (MSE): {np.mean(clf_losses):.6f}")

    print("--- Pre-Training Complete ---")

    # 4. Save Anchored Models
    os.makedirs(pretrain_config['model_save_path'], exist_ok=True)
    cbf_path = os.path.join(pretrain_config['model_save_path'], pretrain_config['cbf_model_name'])
    clf_path = os.path.join(pretrain_config['model_save_path'], pretrain_config['clf_model_name'])
    
    torch.save(cbf_net.state_dict(), cbf_path)
    torch.save(clf_net.state_dict(), clf_path)
    print(f"Saved anchored models to {cbf_path} and {clf_path}")

    # 5. (Optional) Visualize results
    print("Generating visualization of pre-trained critics...")
    env_vis = WarehouseEnv(config)
    goal_vis = np.array(config['fsm']['goal_state'])
    plot_critic_landscapes(
        env=env_vis,
        cbf_net=cbf_net,
        clf_net=clf_net,
        goal=goal_vis,
        device=device,
        filename="visualizations/01_pretrain_landscapes.png"
    )

if __name__ == "__main__":
    os.makedirs("visualizations", exist_ok=True) # Ensure dir exists
    main()