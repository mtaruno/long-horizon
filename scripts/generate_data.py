import yaml
import numpy as np
import os
from tqdm import tqdm

from src.environment.warehouse import WarehouseEnv
from src.utils.buffer import ReplayBuffer
from src.utils.seeding import set_seed
from src.planning.fsm_planner import FSMAutomaton # To get subgoals

def main():
    set_seed(42)
    
    with open("config/warehouse_v1.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    gen_config = config['data_gen']
    nn_config = config['nn']
    
    env = WarehouseEnv(config)
    fsm = FSMAutomaton(
        start_pos=np.array(config['fsm']['start_state']),
        goal_pos=np.array(config['fsm']['goal_state']),
        config=config
    )
    
    buffer = ReplayBuffer(
        state_dim=nn_config['state_dim'],
        action_dim=nn_config['action_dim'],
        subgoal_dim=nn_config['subgoal_dim'],
        max_size=gen_config['num_transitions']
    )
    
    # Get all subgoals from the FSM (works with any number of waypoints)
    # The CLF needs to learn feasibility for all subgoals in the FSM
    subgoals_list = list(fsm.subgoals.values())
    
    print(f"--- Starting Data Generation (Algorithm 3) ---")
    print(f"Targeting {gen_config['num_transitions']} transitions.")
    print(f"Using {len(subgoals_list)} subgoals from FSM: {list(fsm.subgoals.keys())}")
    
    pbar = tqdm(total=gen_config['num_transitions'])
    
    while len(buffer) < gen_config['num_transitions']:
        
        # Convert list to np.array before dividing
        probabilities = np.array([
            gen_config['sampling_mix']['uniform'], 
            gen_config['sampling_mix']['boundary']
        ])
        probabilities /= (gen_config['sampling_mix']['uniform'] + gen_config['sampling_mix']['boundary'])
        
        strategy = np.random.choice(
            ['uniform', 'boundary'], 
            p=probabilities
        )
        
        if strategy == 'uniform':
            internal_state = env.sample_random_state()
        else: # 'boundary'
            internal_state = env.sample_random_state(near_boundary_of=0.5)
        
        env.state = internal_state # Manually set env state
        s_nn = env.get_nn_state(internal_state)
        
        a = env.sample_random_action()
        
        s_next_nn, _, _, info = env.step(a)
        
        # Randomly pick one of the subgoals for this data point
        # This ensures the CLF learns feasibility for all subgoals in the FSM
        # The CLF V_ψ(s, g) needs diverse (s, g) pairs to learn properly
        idx = int(np.random.randint(0, len(subgoals_list)))
        g = subgoals_list[idx]
        
        h_star = info['h_star'] # distance to nearest obstacle
        v_star = env.get_ground_truth_feasibility(s_next_nn, g) # distance to subgoal g
        
        buffer.add(
            state=s_nn,
            action=a,
            next_state=s_next_nn,
            subgoal=g,
            reward=0.0,
            done=info['is_collision'],
            h_star=h_star,
            v_star=v_star
        )
        pbar.update(1)

    pbar.close()
    
    os.makedirs(os.path.dirname(gen_config['data_path']), exist_ok=True)
    buffer.save(gen_config['data_path'])
    print(f"\n--- Data Generation Complete ---")
    print(f"Saved {len(buffer)} transitions to {gen_config['data_path']}.")

if __name__ == "__main__":
    main()