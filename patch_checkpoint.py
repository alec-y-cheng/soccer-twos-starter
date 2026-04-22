import pickle
import os
import sys

def patch_checkpoint(input_path, output_path):
    print(f"Opening checkpoint at {input_path}...")
    with open(input_path, 'rb') as f:
        data = pickle.load(f)

    # Decode the 'worker' binary data
    worker = pickle.loads(data['worker'])
    
    patched_any = False
    new_state = {}
    new_filters = {}
    
    for policy_name, policy_state in worker['state'].items():
        if isinstance(policy_state, dict):
            # Map multi-agent 'default' back to standard 'default_policy' for local rendering compatibility
            new_name = 'default_policy' if policy_name == 'default' else policy_name
            
            if 'weights' not in policy_state:
                print(f"Structural mismatch detected for policy: '{policy_name}'. Patching to '{new_name}'...")
                # 1. Filter out optimizer variables
                weights_dict = {k: v for k, v in policy_state.items() if k != '_optimizer_variables'}
                
                # 2. Reshape into expected dictionary with global_timestep
                new_state[new_name] = {
                    'weights': weights_dict,
                    'global_timestep': 0
                }
                patched_any = True
            else:
                new_state[new_name] = policy_state
                
            # Carry over identically named filters if they exist
            if 'filters' in worker and policy_name in worker['filters']:
                new_filters[new_name] = worker['filters'][policy_name]
        else:
            new_state[policy_name] = policy_state

    if patched_any:
        worker['state'] = new_state
        if 'filters' in worker:
            worker['filters'] = new_filters
            
        # 3. Recode worker back into binary
        data['worker'] = pickle.dumps(worker)
        
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Successfully patched and saved as: {output_path}")
    else:
        print("No structural mismatches found (no patching needed).")

if __name__ == "__main__":
    # You can change these paths as needed
    INPUT = 'checkpoints/best/checkpoint-2479'
    OUTPUT = INPUT
    patch_checkpoint(INPUT, OUTPUT)
