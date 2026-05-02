import os
import pickle
import torch
import numpy as np

def export_weights(checkpoint_path, output_dir):
    print(f"Loading checkpoint from {checkpoint_path}...")
    with open(checkpoint_path, "rb") as f:
        data = pickle.load(f)
    
    worker = pickle.loads(data["worker"])
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Process Striker
    if "striker" in worker["state"]:
        print("Extracting Striker weights...")
        striker_state = worker["state"]["striker"]
        # In multi-agent RLLib torch, weights are in 'weights' key
        weights = striker_state["weights"] if "weights" in striker_state else striker_state
        
        # We need to map keys:
        # '_hidden_layers.0._model.0.weight' -> 'hidden.0.weight'
        # '_logits_layer._model.0.weight' -> 'logits.weight'
        
        clean_striker = {}
        # Mapping for 2 hidden layers
        clean_striker["hidden.0.weight"] = torch.from_numpy(weights["_hidden_layers.0._model.0.weight"])
        clean_striker["hidden.0.bias"] = torch.from_numpy(weights["_hidden_layers.0._model.0.bias"])
        clean_striker["hidden.2.weight"] = torch.from_numpy(weights["_hidden_layers.1._model.0.weight"])
        clean_striker["hidden.2.bias"] = torch.from_numpy(weights["_hidden_layers.1._model.0.bias"])
        clean_striker["logits.weight"] = torch.from_numpy(weights["_logits._model.0.weight"])
        clean_striker["logits.bias"] = torch.from_numpy(weights["_logits._model.0.bias"])
        
        torch.save(clean_striker, os.path.join(output_dir, "striker.pth"))
        print(f"Saved striker.pth to {output_dir}")

    # Process Goalie
    if "goalie" in worker["state"]:
        print("Extracting Goalie weights...")
        goalie_state = worker["state"]["goalie"]
        weights = goalie_state["weights"] if "weights" in goalie_state else goalie_state
        
        clean_goalie = {}
        clean_goalie["hidden.0.weight"] = torch.from_numpy(weights["_hidden_layers.0._model.0.weight"])
        clean_goalie["hidden.0.bias"] = torch.from_numpy(weights["_hidden_layers.0._model.0.bias"])
        clean_goalie["hidden.2.weight"] = torch.from_numpy(weights["_hidden_layers.1._model.0.weight"])
        clean_goalie["hidden.2.bias"] = torch.from_numpy(weights["_hidden_layers.1._model.0.bias"])
        clean_goalie["logits.weight"] = torch.from_numpy(weights["_logits._model.0.weight"])
        clean_goalie["logits.bias"] = torch.from_numpy(weights["_logits._model.0.bias"])
        
        torch.save(clean_goalie, os.path.join(output_dir, "goalie.pth"))
        print(f"Saved goalie.pth to {output_dir}")

    # --- NEW: Copy the Boilerplate Code ---
    import shutil
    # The parent directory of 'model/' is the package root
    package_root = os.path.dirname(output_dir)
    source_dir = "submission"
    
    files_to_copy = ["agent_ray.py", "model.py", "utils.py", "__init__.py"]
    print(f"Copying boilerplate code to {package_root}...")
    
    for f in files_to_copy:
        src = os.path.join(source_dir, f)
        dst = os.path.join(package_root, f)
        if os.path.exists(src):
            shutil.copy(src, dst)
        else:
            print(f"Warning: Could not find {src} to copy!")
    
    print(f"✓ Full submission package created at: {package_root}")

if __name__ == "__main__":
    # Example usage:
    # This will create a folder 'BetterStrikerRewards' containing everything 
    # needed for submission (Code + Weights).
    CHECKPOINT = "ray_results/PPO_league_training/PPO_Soccer_2926d_00000_0_2026-04-21_13-04-30/checkpoint_001900/checkpoint-1900"
    export_weights(CHECKPOINT, "WasThisBest/model")
