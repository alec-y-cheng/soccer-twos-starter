import pickle
import os
import sys

def patch_checkpoint(input_path, output_path):
    print(f"Opening checkpoint at {input_path}...")
    with open(input_path, 'rb') as f:
        data = pickle.load(f)

    # Decode the 'worker' binary data
    worker = pickle.loads(data['worker'])
    
    if 'default_policy' in worker['state']:
        policy_state = worker['state']['default_policy']
        
        # Check if it needs patching
        if 'weights' not in policy_state:
            print("Structural mismatch detected. Patching...")
            # 1. Filter out optimizer variables
            weights_dict = {k: v for k, v in policy_state.items() if k != '_optimizer_variables'}
            
            # 2. Reshape into expected dictionary with global_timestep
            worker['state']['default_policy'] = {
                'weights': weights_dict,
                'global_timestep': 0
            }
            
            # 3. Recode worker back into binary
            data['worker'] = pickle.dumps(worker)
            
            with open(output_path, 'wb') as f:
                pickle.dump(data, f)
            
            print(f"Successfully patched and saved as: {output_path}")
        else:
            print("Checkpoint already has correct structure (no patching needed).")
    else:
        print("Error: 'default_policy' was not found in checkpoint state.")

if __name__ == "__main__":
    # You can change these paths as needed
    INPUT = 'checkpoint_001514/checkpoint-1514'
    OUTPUT = 'checkpoint_001514/checkpoint-1514-patched'
    
    patch_checkpoint(INPUT, OUTPUT)
