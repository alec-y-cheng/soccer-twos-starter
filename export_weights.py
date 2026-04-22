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
        clean_striker["hidden.0.weight"] = weights["_hidden_layers.0._model.0.weight"]
        clean_striker["hidden.0.bias"] = weights["_hidden_layers.0._model.0.bias"]
        clean_striker["hidden.2.weight"] = weights["_hidden_layers.1._model.0.weight"]
        clean_striker["hidden.2.bias"] = weights["_hidden_layers.1._model.0.bias"]
        clean_striker["logits.weight"] = weights["_logits._model.0.weight"]
        clean_striker["logits.bias"] = weights["_logits._model.0.bias"]
        
        torch.save(clean_striker, os.path.join(output_dir, "striker.pth"))
        print(f"Saved striker.pth to {output_dir}")

    # Process Goalie
    if "goalie" in worker["state"]:
        print("Extracting Goalie weights...")
        goalie_state = worker["state"]["goalie"]
        weights = goalie_state["weights"] if "weights" in goalie_state else goalie_state
        
        clean_goalie = {}
        clean_goalie["hidden.0.weight"] = weights["_hidden_layers.0._model.0.weight"]
        clean_goalie["hidden.0.bias"] = weights["_hidden_layers.0._model.0.bias"]
        clean_goalie["hidden.2.weight"] = weights["_hidden_layers.1._model.0.weight"]
        clean_goalie["hidden.2.bias"] = weights["_hidden_layers.1._model.0.bias"]
        clean_goalie["logits.weight"] = weights["_logits._model.0.weight"]
        clean_goalie["logits.bias"] = weights["_logits._model.0.bias"]
        
        torch.save(clean_goalie, os.path.join(output_dir, "goalie.pth"))
        print(f"Saved goalie.pth to {output_dir}")

if __name__ == "__main__":
    CHECKPOINT = "ray_results/PPO_league_training/PPO_Soccer_2926d_00000_0_2026-04-21_13-04-30/checkpoint_001900/checkpoint-1900"
    export_weights(CHECKPOINT, "submission/model")
