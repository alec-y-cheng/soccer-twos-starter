import numpy as np
import ray
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
from utils import create_rllib_env


NUM_ENVS_PER_WORKER = 3

def policy_mapping_fn(agent_id, episode=None, worker=None, **kwargs):
    """
    Team A (Agents 0 and 1) is always our learning duo.
    Team B (Agents 2 and 3) rotates opponents based on episode ID.
    PID 0, 2 = Striker. PID 1, 3 = Goalie.
    """
    if agent_id == 0:
        return "striker"
    elif agent_id == 1:
        return "goalie"
        
    # If episode metadata isn't available during init, provide a default
    if episode is None:
        return "baseline"
        
    # We use episode_id to ensure Agent 2 and Agent 3 in the same match 
    # agree on which opponent style they are currently playing.
    np.random.seed(episode.episode_id)
    opponent_type = np.random.choice(["self", "baseline", "dummy"], p=[0.4, 0.4, 0.2])
    
    if opponent_type == "self":
        # Mirror match against another copy of the learning team
        if agent_id == 2: return "striker"
        if agent_id == 3: return "goalie"
    elif opponent_type == "baseline":
        return "baseline" # Baseline agent plays both physical roles with 1 generalized brain
    elif opponent_type == "dummy":
        return "dummy" # A completely random untrained agent

class PolicyWeightsCallback(DefaultCallbacks):
    def on_trainer_init(self, *, trainer, **kwargs):
        striker_path = "ray_results/reward_wrapper_test/PPO_Soccer_e0b85_00000_0_2026-04-20_13-53-46/checkpoint_002479/checkpoint-2479"
        goalie_path = "ray_results/PPO_selfplay_goalie/PPO_Soccer_68a2c_00000_0_2026-04-21_02-36-21/checkpoint_001008/checkpoint-1008"
        baseline_path = "ceia_baseline_agent/ray_results/PPO_selfplay_twos/PPO_Soccer_f475e_00000_0_2021-09-19_15-54-02/checkpoint_002449/checkpoint-2449"
        
        import pickle
        
        # Load Striker
        try:
            with open(striker_path, "rb") as f: data = pickle.load(f)
            worker_state = pickle.loads(data["worker"])
            
            # Accommodate whether it was saved under "default" or "striker"
            weights = worker_state["policy_map"].get("striker", worker_state["policy_map"].get("default"))
            trainer.get_policy("striker").set_state(weights)
            print("✅ Loaded Striker successfully.")
        except Exception as e: 
            print(f"❌ Striker load skipped/error: {e}")

        # Load Goalie
        try:
            with open(goalie_path, "rb") as f: data = pickle.load(f)
            worker_state = pickle.loads(data["worker"])
            weights = worker_state["policy_map"]["goalie"]
            trainer.get_policy("goalie").set_state(weights)
            print("✅ Loaded Goalie successfully.")
        except Exception as e: 
            print(f"❌ Goalie load skipped/error: {e}")

        # Load Baseline
        try:
            with open(baseline_path, "rb") as f: data = pickle.load(f)
            worker_state = pickle.loads(data["worker"])
            weights = worker_state["policy_map"]["default"] # Baseline used 'default'
            trainer.get_policy("baseline").set_state(weights)
            print("✅ Loaded CEIA Baseline successfully.")
        except Exception as e: 
            print(f"❌ Baseline load skipped/error: {e}")


if __name__ == "__main__":
    ray.init(include_dashboard=False, _node_ip_address="0.0.0.0")

    tune.registry.register_env("Soccer", create_rllib_env)
    temp_env = create_rllib_env()
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space
    temp_env.close()

    tune.run(
        "PPO",
        name="PPO_league_training",
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
                    "striker": (None, obs_space, act_space, {}),
                    "goalie": (None, obs_space, act_space, {}),
                    "baseline": (None, obs_space, act_space, {}),
                    "dummy": (None, obs_space, act_space, {}),
                },
                "policy_mapping_fn": policy_mapping_fn,
                "policies_to_train": ["striker", "goalie"], # Baseline and Dummy are frozen!
            },
            "env": "Soccer",
            "env_config": {
                "num_envs_per_worker": NUM_ENVS_PER_WORKER,
                "reward_shaping": "team",  # Use the powerful new wrapper we built
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
        stop={"timesteps_total": 50000000, "time_total_s": 43200,},  # 12h
        checkpoint_freq=100,
        checkpoint_at_end=True,
        local_dir="./ray_results",
    )
