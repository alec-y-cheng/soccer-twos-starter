import soccer_twos
import numpy as np

env = soccer_twos.make(single_player=False)
obs = env.reset()
action_dict = {0: [0,0,0], 1: [0,0,0], 2: [0,0,0], 3: [0,0,0]}
obs, reward, done, info = env.step(action_dict)

print("\n--- Positions ---")
for pid in info.keys():
    print(f"Agent {pid}: {info[pid]['player_info']['position']}")
print(f"Ball: {info[0]['ball_info']['position']}")
