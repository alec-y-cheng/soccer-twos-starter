import ray
from ray.rllib.agents.ppo import PPOTrainer
from soccer_twos import EnvType
from ray import tune

from utils import create_rllib_env

if __name__ == "__main__":
    ray.init()
    
    # We register the custom Soccer environment with RLlib
    tune.registry.register_env("Soccer", create_rllib_env)
    
    # Define the config that matches the checkpoint (assuming it came from train_ray_curriculum.py)
    # If the checkpoint came from a different script, adjust the 'env_config' and 'model' accordingly.
    config = {
        "num_gpus": 0,
        "num_workers": 1,
        "framework": "torch",
        "env": "Soccer",
        "env_config": {
            "num_envs_per_worker": 1,
            "variation": EnvType.team_vs_policy,
            "multiagent": False,
            "flatten_branched": True,
            "single_player": True,
            "opponent_policy": lambda *_: 0,
            "watch": True, # Enables visualization by opening the Unity GUI
        },
        "model": {
            "vf_share_layers": True,
            "fcnet_hiddens": [256, 256],
            "fcnet_activation": "relu",
        },
        "explore": False, # Turn off exploration for evaluation
    }
    
    # Instantiate the trainer and load the copied checkpoint
    trainer = PPOTrainer(config=config, env="Soccer")
    checkpoint_path = "checkpoint_098/checkpoint-98-patched"
    print(f"Restoring from {checkpoint_path}...")
    trainer.restore(checkpoint_path)
    
    # Create the environment for evaluation loop
    env = create_rllib_env(config["env_config"])
    
    obs = env.reset()
    done = False
    total_reward = 0
    print("Starting evaluation episode...")
    
    # Run the environment evaluation loop
    while not done:
        action = trainer.compute_single_action(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        
    print(f"Episode Done! Total Reward: {total_reward}")
    env.close()
    ray.shutdown()
