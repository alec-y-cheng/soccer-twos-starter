import numpy as np
import ray
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
from utils import create_rllib_env


NUM_ENVS_PER_WORKER = 3

def policy_mapping_fn(agent_id, episode=None, worker=None, **kwargs):
    """
    Team A (Agents 0 and 1) is always our learning duo.
    Team B (Agents 2 and 3) rotates opponents with a focus on Self-Play "Boss" matches.
    """
    if agent_id == 0:
        return "striker"
    elif agent_id == 1:
        return "goalie"
        
    if episode is None:
        return "baseline"
        
    # Phase 2: Shift focus to Self-Play as agents are now very smart.
    np.random.seed(episode.episode_id)
    opponent_pool = np.random.choice(["self", "baseline", "dummy"], p=[0.7, 0.2, 0.1])
    
    if opponent_pool == "self":
        # 15% chance to face "Aggressive" mode (2 Strikers) to stress-test the goalie
        # 85% chance for standard balanced team
        is_aggressive = np.random.random() < 0.15
        if is_aggressive:
            return "striker" # Both 2 and 3 become strikers
        else:
            if agent_id == 2: return "striker"
            if agent_id == 3: return "goalie"
    elif opponent_pool == "baseline":
        return "baseline"
    elif opponent_pool == "dummy":
        return "dummy"

class PolicyWeightsCallback(DefaultCallbacks):
    def on_trainer_init(self, *, trainer, **kwargs):
        # RESUMING FROM 10-HOUR LEAGUE RUN
        # ray_results/PPO_league_training/PPO_Soccer_4ec4b_00000_0_2026-04-22_02-48-45/checkpoint_001800/checkpoint-1800
        # this recent one is trained on itself but the old rewards, the older one is trained on better striker rewards (1-34)
        master_path = "ray_results/PPO_league_training/PPO_Soccer_4ec4b_00000_0_2026-04-22_02-48-45/checkpoint_001800/checkpoint-1800"
        baseline_path = "ray_results/PPO_league_training/PPO_Soccer_4ec4b_00000_0_2026-04-22_02-48-45/checkpoint_001800/checkpoint-1800"
        
        import pickle
        
        # Load unified Striker and Goalie from the same master file
        try:
            with open(master_path, "rb") as f: data = pickle.load(f)
            worker_state = pickle.loads(data["worker"])
            
            # Pull Striker weights
            s_weights = worker_state["policy_map"]["striker"]
            trainer.get_policy("striker").set_state(s_weights)
            
            # Pull Goalie weights
            g_weights = worker_state["policy_map"]["goalie"]
            trainer.get_policy("goalie").set_state(g_weights)
            
            print(f"✅ Successfully resumed Striker & Goalie from {master_path}")
        except Exception as e: 
            print(f"❌ Resume failed: {e}")

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
        name="PPO_league_training_better_striker_rewards",
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
