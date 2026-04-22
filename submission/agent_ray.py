import os
import torch
import numpy as np
from soccer_twos import AgentInterface

# Import our standalone model
from .model import SimplePolicyNetwork

class Agent(AgentInterface):
    """
    Stand-alone agent that uses Pure PyTorch for inference.
    Handles both Striker and Goalie roles in a 2v2 environment.
    """
    def __init__(self, env=None):
        self.name = "RayPPO-League-Agent"
        # Initialize Striker and Goalie brains
        self.striker_model = SimplePolicyNetwork(obs_size=336, action_dims=[3, 3, 3])
        self.goalie_model = SimplePolicyNetwork(obs_size=336, action_dims=[3, 3, 3])
        
        # Load weights
        model_dir = os.path.join(os.path.dirname(__file__), "model")
        striker_path = os.path.join(model_dir, "striker.pth")
        goalie_path = os.path.join(model_dir, "goalie.pth")
        
        if os.path.exists(striker_path):
            self.striker_model.load_state_dict(torch.load(striker_path))
            print("✓ Loaded Striker weights.")
        
        if os.path.exists(goalie_path):
            self.goalie_model.load_state_dict(torch.load(goalie_path))
            print("✓ Loaded Goalie weights.")
            
        self.striker_model.eval()
        self.goalie_model.eval()

    def act(self, observation):
        """
        Receives a dict of {pid: obs} and returns a dict of {pid: action}.
        0, 2 = Striker
        1, 3 = Goalie
        """
        actions = {}
        for pid, obs in observation.items():
            # Convert obs to tensor [1, 336]
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
            
            # Select the correct brain based on PID
            if pid in [0, 2]:
                actions[pid] = self.striker_model.get_action(obs_tensor)
            else:
                actions[pid] = self.goalie_model.get_action(obs_tensor)
                
        return actions
