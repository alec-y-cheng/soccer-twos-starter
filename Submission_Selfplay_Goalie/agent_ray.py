
import os
import torch
import numpy as np
from .model import SimpleFCNet
from .utils import SimpleObservationWrapper

class SubmissionAgent:
    def __init__(self):
        self.name = "Selfplay_Goalie"
        # Determine observation dim
        obs_dim = 4 if False else 336
        
        self.striker = SimpleFCNet(obs_dim, 3)
        self.goalie = SimpleFCNet(obs_dim, 3)
        
        # Load weights if they exist
        s_path = os.path.join(os.path.dirname(__file__), "striker.pt")
        g_path = os.path.join(os.path.dirname(__file__), "goalie.pt")
        
        if os.path.exists(s_path):
            self.striker.load_state_dict(torch.load(s_path))
        if os.path.exists(g_path):
            self.goalie.load_state_dict(torch.load(g_path))
        elif os.path.exists(s_path):
            self.goalie.load_state_dict(torch.load(s_path)) # Fallback

    def act(self, observation):
        # observation is a dict of {pid: obs}
        actions = {}
        for pid, obs in observation.items():
            # Apply wrapper if simple
            if False:
                # Hand-crank the wrapper logic for a single obs
                # (SimpleObservationWrapper usually wraps the env, but we need it for inference)
                # For simplicity in this bundle, we assume the wrapper is already integrated or handled
                pass
            
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
            if pid in [0, 2]:
                action_logits, _ = self.striker({"obs": obs_tensor})
            else:
                action_logits, _ = self.goalie({"obs": obs_tensor})
            
            # Simple argmax for MultiDiscrete
            # action_logits is shape [1, 9] (3x3 multidiscrete flattened)
            # But SimpleFCNet returns [1, 9] usually.
            # We need to split into 3, 3, 3
            logits = action_logits.detach().numpy()[0]
            actions[pid] = [
                np.argmax(logits[0:3]),
                np.argmax(logits[3:6]),
                np.argmax(logits[6:9])
            ]
        return actions
