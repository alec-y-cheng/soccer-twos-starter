
import os
import torch
import numpy as np
from .model import SimpleFCNet
from .utils import SimpleObservationWrapper

class SubmissionAgent:
    def __init__(self):
        self.name = "Priv_Info_Simple"
        # Determine observation dim
        obs_dim = 4 if True else 336
        
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
            # Apply compression for Priv_Info_Simple agent
            # Extract Goal Hits [1::8] and Distances [7::8]
            goal_hits = obs[1::8]
            distances = obs[7::8]
            num_rays = len(goal_hits)
            
            goal_x, goal_y = 0.0, 0.0
            for i in range(num_rays):
                angle = (i / num_rays) * 2 * np.pi
                dist = distances[i]
                if goal_hits[i] == 1.0:
                    goal_x = dist * np.cos(angle)
                    goal_y = dist * np.sin(angle)
            
            # For this agent, ball information is already part of the raycasts 
            # if we use the 'simple' logic from training.
            # However, the training script used the 'SimpleObservationWrapper'
            # which pulled absolute ball pos from the 'info' dict.
            # Since 'info' isn't available in act(), we fallback to raycast-based ball detection.
            ball_hits = obs[0::8]
            ball_x, ball_y = 0.0, 0.0
            for i in range(num_rays):
                angle = (i / num_rays) * 2 * np.pi
                dist = distances[i]
                if ball_hits[i] == 1.0:
                    ball_x = dist * np.cos(angle)
                    ball_y = dist * np.sin(angle)

            compressed_obs = np.array([ball_x, ball_y, goal_x, goal_y], dtype=np.float32)
            obs_tensor = torch.from_numpy(compressed_obs).float().unsqueeze(0)
            
            if pid in [0, 2]:
                action_logits, _ = self.striker({"obs": obs_tensor})
            else:
                action_logits, _ = self.goalie({"obs": obs_tensor})
            
            logits = action_logits.detach().numpy()[0]
            actions[pid] = [
                np.argmax(logits[0:3]),
                np.argmax(logits[3:6]),
                np.argmax(logits[6:9])
            ]
        return actions
