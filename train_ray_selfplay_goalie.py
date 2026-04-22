import numpy as np
import ray
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
from utils import create_rllib_env


NUM_ENVS_PER_WORKER = 3


def policy_mapping_fn(agent_id, *args, **kwargs):
    # Agents 0 and 2 are usually Strikers (closer to center) -> map to "default" (from checkpoint)
    # Agents 1 and 3 are usually Goalies (closer to net)     -> map to "goalie"
    if agent_id in [0, 2]:
        return "default"
    else:
        return "goalie"

class PolicyWeightsCallback(DefaultCallbacks):
    def on_trainer_init(self, *, trainer, **kwargs):
        checkpoint_path = "ray_results/reward_wrapper_test/PPO_Soccer_e0b85_00000_0_2026-04-20_13-53-46/checkpoint_002479/checkpoint-2479"
        print(f"Surgically loading Striker weights from {checkpoint_path}")
        
        try:
            # We only want to load the "default" policy weights from the checkpoint
            # and inject them into our local "default" policy.
            import pickle
            with open(checkpoint_path, "rb") as f:
                checkpoint_data = pickle.load(f)
            
            # Extract weights for the 'default' policy specifically
            worker_state = pickle.loads(checkpoint_data["worker"])
            striker_weights = worker_state["policy_map"]["default"]
            
            # Apply them
            trainer.get_policy("default").set_state(striker_weights)
            print("Successfully injected Striker weights!")
        except Exception as e:
            print(f"Warning: Could not load weights surgically: {e}")
            print("Training will continue with fresh weights for both.")

# Removed SelfPlayUpdateCallback since we aren't league training opponents here


if __name__ == "__main__":
    # include_dashboard=False prevents Ray from trying to bind a dashboard port,
    # which fails on SLURM compute nodes. _node_ip_address forces Ray to bind
    # to localhost instead of trying to resolve the node's external hostname.
    ray.init(include_dashboard=False, _node_ip_address="0.0.0.0")

    tune.registry.register_env("Soccer", create_rllib_env)
    temp_env = create_rllib_env()
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space
    temp_env.close()

    analysis = tune.run(
        "PPO",
        name="PPO_selfplay_goalie",
        config={
            # system settings
            "num_gpus": 1,
            "num_workers": 8,
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "log_level": "INFO",
            "framework": "torch",
            "callbacks": PolicyWeightsCallback,
            # RL setup
            "multiagent": {
                "policies": {
                    "default": (None, obs_space, act_space, {}), # Restored from checkpoint
                    "goalie": (None, obs_space, act_space, {}),  # Freshly created for training
                },
                "policy_mapping_fn": tune.function(policy_mapping_fn),
                "policies_to_train": ["goalie"], # Train goalie, freeze default (striker)!
            },
            "env": "Soccer",
            "env_config": {
                "num_envs_per_worker": NUM_ENVS_PER_WORKER,
                "reward_shaping": "goalie",  # Use the Geometric Goalie reward logic!
                "obs_type": "raw",           
            },
            "model": {
                "vf_share_layers": True,
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "relu",
            },
            "rollout_fragment_length": 5000,
            "batch_mode": "complete_episodes",
        },
        stop={"timesteps_total": 15000000, "time_total_s": 28800,},  # 8h
        checkpoint_at_end=True,
        local_dir="./ray_results",
        # NOTE: We no longer use 'restore' here because it causes Multi-Agent mismatches.
        # The weights are now loaded surgically via PolicyWeightsCallback.
    )

    # Gets best trial based on max accuracy across all training iterations.
    best_trial = analysis.get_best_trial("episode_reward_mean", mode="max")
    print(best_trial)
    # Gets best checkpoint for trial based on accuracy.
    best_checkpoint = analysis.get_best_checkpoint(
        trial=best_trial, metric="episode_reward_mean", mode="max"
    )
    print(best_checkpoint)
    print("Done training")
