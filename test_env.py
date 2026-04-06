import soccer_twos
import numpy as np

# Create a random env 
env = soccer_twos.make(single_player=True)

# Start the game
obs = env.reset()

print("OBSERVATION TYPE:", type(obs))
for player_id, player_obs in obs.items():
    print(f"PLAYER {player_id} SHAPE: {np.array(player_obs).shape}")
    print(f"PLAYER {player_id} DATA:", player_obs)
