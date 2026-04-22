import soccer_twos
import numpy as np

# Use absolute import logic because this is the package main
from submission import Agent

def run_local_match():
    print("Initializing Soccer Twos environment for local test...")
    # By default, use Team vs Team mode to see all 4 players
    env = soccer_twos.make(render=True, variation=soccer_twos.EnvType.multiagent_player)
    
    agent = Agent(env)
    print(f"Agent '{agent.name}' initialized. Starting match...")
    
    obs = env.reset()
    done = {"__all__": False}
    total_reward = 0
    
    try:
        while not done["__all__"]:
            # Our Agent handles the dict of {pid: obs} and returns {pid: action}
            actions = agent.act(obs)
            obs, reward, done, info = env.step(actions)
            total_reward += sum(reward.values())
        
        print(f"Match Finished. Total combined reward: {total_reward:.3f}")
    except KeyboardInterrupt:
        print("\nMatch stopped by user.")
    finally:
        env.close()

if __name__ == "__main__":
    run_local_match()
