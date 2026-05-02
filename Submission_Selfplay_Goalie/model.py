import torch
import torch.nn as nn

class SimplePolicyNetwork(nn.Module):
    """
    A standalone PyTorch implementation of the RLlib FCNet architecture.
    Matches: [256, 256] hiddens, ReLU activation.
    """
    def __init__(self, obs_size=336, action_dims=[3, 3, 3]):
        super().__init__()
        self.action_dims = action_dims
        
        # Hidden layers
        self.hidden = nn.Sequential(
            nn.Linear(obs_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        
        # Policy head (logits)
        self.logits = nn.Linear(256, sum(action_dims))

    def forward(self, x):
        x = self.hidden(x)
        return self.logits(x)

    def get_action(self, obs_tensor):
        """
        Calculates the argmax action for MultiDiscrete [3, 3, 3]
        """
        with torch.no_grad():
            logits = self.forward(obs_tensor)
            # Split logits into chunks of 3 and take argmax of each
            actions = []
            curr = 0
            for dim in self.action_dims:
                chunk = logits[:, curr:curr+dim]
                actions.append(torch.argmax(chunk, dim=1).item())
                curr += dim
            return actions
