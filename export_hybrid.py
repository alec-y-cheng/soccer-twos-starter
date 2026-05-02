import os
import pickle
import torch
import numpy as np
import shutil

def extract_policy_weights(checkpoint_path, policy_name):
    print(f"Loading {policy_name} from {checkpoint_path}...")
    with open(checkpoint_path, "rb") as f:
        data = pickle.load(f)
    worker = pickle.loads(data["worker"])
    
    if policy_name not in worker["state"]:
        # Fallback to 'default' if the explicit name isn't there
        print(f"  Warning: '{policy_name}' not found, trying 'default'...")
        policy_state = worker["state"]["default"]
    else:
        policy_state = worker["state"][policy_name]
        
    weights = policy_state["weights"] if "weights" in policy_state else policy_state
    
    clean_weights = {}
    # Mapping for 2 hidden layers
    clean_weights["hidden.0.weight"] = torch.from_numpy(weights["_hidden_layers.0._model.0.weight"])
    clean_weights["hidden.0.bias"] = torch.from_numpy(weights["_hidden_layers.0._model.0.bias"])
    clean_weights["hidden.2.weight"] = torch.from_numpy(weights["_hidden_layers.1._model.0.weight"])
    clean_weights["hidden.2.bias"] = torch.from_numpy(weights["_hidden_layers.1._model.0.bias"])
    clean_weights["logits.weight"] = torch.from_numpy(weights["_logits._model.0.weight"])
    clean_weights["logits.bias"] = torch.from_numpy(weights["_logits._model.0.bias"])
    
    return clean_weights

def export_hybrid(striker_checkpoint, goalie_checkpoint, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Extract Striker
    striker_weights = extract_policy_weights(striker_checkpoint, "striker")
    torch.save(striker_weights, os.path.join(output_dir, "striker.pth"))
    print(f"✓ Saved striker.pth to {output_dir}")
    
    # 2. Extract Goalie
    goalie_weights = extract_policy_weights(goalie_checkpoint, "goalie")
    torch.save(goalie_weights, os.path.join(output_dir, "goalie.pth"))
    print(f"✓ Saved goalie.pth to {output_dir}")
    
    # 3. Copy Boilerplate Code
    package_root = os.path.dirname(output_dir)
    source_dir = "submission"
    files_to_copy = ["agent_ray.py", "model.py", "utils.py", "__init__.py"]
    
    print(f"Copying boilerplate code to {package_root}...")
    for f in files_to_copy:
        src = os.path.join(source_dir, f)
        dst = os.path.join(package_root, f)
        if os.path.exists(src):
            shutil.copy(src, dst)
            
    print(f"✓✓ FULL HYBRID PACKAGE CREATED AT: {package_root}")

if __name__ == "__main__":
    STRIKER_CKPT = "ray_results/PPO_league_training/PPO_Soccer_2926d_00000_0_2026-04-21_13-04-30/checkpoint_001900/checkpoint-1900"
    GOALIE_CKPT = "ray_results/PPO_selfplay_goalie/PPO_Soccer_68a2c_00000_0_2026-04-21_02-36-21/checkpoint_001008/checkpoint-1008"
    
    export_hybrid(STRIKER_CKPT, GOALIE_CKPT, "HybridSubmission/model")
