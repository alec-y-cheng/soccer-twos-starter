import ray
import numpy as np
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from soccer_twos import EnvType

# ── CONFIGURATION TOGGLES ───────────────────────────────────────────────
CHECKPOINT_PATH = 'checkpoints/team/checkpoint_001900/checkpoint-1900'

# Options: "raw" (336), "simple" (4), "extended" (10)
OBS_TYPE = "raw" 

# True = Discrete(27), False = MultiDiscrete([3,3,3])
FLATTENED = False

# ── TEAM MODE: Set to True if checkpoint has separate Striker/Goalie policies
TEAM_MODE = True

# ── DUAL CONTROL: If True, both players on Blue are driven by policy
CONTROL_BOTH_PLAYERS = True

# ── BASELINE: If True, Orange team is driven by CEIA Baseline
PLAY_AGAINST_BASELINE = False
BASELINE_PATH = 'ceia_baseline_agent/ray_results/PPO_selfplay_twos/PPO_Soccer_f475e_00000_0_2021-09-19_15-54-02/checkpoint_002449/checkpoint-2449-patched'

from utils import create_rllib_env

def get_config(observation_space, action_space):
    config = {
        "num_gpus": 0, "num_workers": 0, "framework": "torch", "explore": False,
        "env": "DummyEnv", "disable_env_checking": True,
        "observation_space": observation_space, "action_space": action_space,
        "model": {"vf_share_layers": True, "fcnet_hiddens": [256, 256], "fcnet_activation": "relu"},
    }
    
    if TEAM_MODE:
        config["multiagent"] = {
            "policies": {
                "striker": (None, observation_space, action_space, {}),
                "goalie": (None, observation_space, action_space, {}),
                "baseline": (None, observation_space, action_space, {}),
                "dummy": (None, observation_space, action_space, {}),
            },
            "policy_mapping_fn": lambda agent_id, **kwargs: "striker" if agent_id in [0, 2] else "goalie",
        }
    return config

if __name__ == "__main__":
    ray.init(include_dashboard=False, _node_ip_address="0.0.0.0")
    
    import gym
    obs_shape = (336 if OBS_TYPE=="raw" else (4 if OBS_TYPE=="simple" else 10),)
    observation_space = gym.spaces.Box(low=-1.0, high=2.0, shape=obs_shape, dtype=np.float32)
    action_space = gym.spaces.Discrete(27) if FLATTENED else gym.spaces.MultiDiscrete([3, 3, 3])

    # Register DummyEnv so trainers don't open Unity
    from ray.rllib.env.multi_agent_env import MultiAgentEnv
    class MAEnv(MultiAgentEnv):
        def __init__(self, *args, **kwargs):
            self.observation_space = observation_space
            self.action_space = action_space
        def reset(self): return {}
        def step(self, action): return {}, {}, {}, {}

    tune.registry.register_env("DummyEnv", lambda _: MAEnv()) 
    tune.registry.register_env("Soccer", create_rllib_env)

    # 1. Initialize your trainer
    trainer = PPOTrainer(config=get_config(observation_space, action_space))
    trainer.restore(CHECKPOINT_PATH)
    print(f"✓ Loaded your checkpoint: {CHECKPOINT_PATH}")

    # 2. Baseline setup
    baseline_trainer = None
    if PLAY_AGAINST_BASELINE:
        # Baseline is always single-policy non-flattened MultiDiscrete([3,3,3])
        baseline_action_space = gym.spaces.MultiDiscrete([3,3,3])
        b_config = get_config(observation_space, baseline_action_space)
        baseline_trainer = PPOTrainer(config=b_config)
        baseline_trainer.restore(BASELINE_PATH)
        print(f"✓ Loaded baseline checkpoint: {BASELINE_PATH}")

    # 3. Environment
    eval_env_config = {
        "variation": EnvType.multiagent_player, # Always use player-mode for team control
        "multiagent": False, # Returns dict obs for all 4 agents
        "flatten_branched": FLATTENED,
        "single_player": False,
        "obs_type": OBS_TYPE,
        "watch": True, "worker_id": 1,
    }
    env = create_rllib_env(eval_env_config)

    obs_len = observation_space.shape[0]

    for episode in range(5):
        obs = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action = {}
            for agent_id, agent_obs in obs.items():
                # --- ORANGE TEAM (2, 3) ---
                if agent_id in [2, 3]:
                    if PLAY_AGAINST_BASELINE:
                        raw_act = baseline_trainer.compute_single_action(agent_obs, explore=False)
                        action[agent_id] = (raw_act[0]*9 + raw_act[1]*3 + raw_act[2]) if FLATTENED else raw_act
                    else:
                        # Built-in bot or random
                        action[agent_id] = env.action_space.sample() 
                
                # --- BLUE TEAM (0, 1) ---
                else:
                    if agent_id == 1 and not CONTROL_BOTH_PLAYERS:
                        action[1] = env.action_space.sample() # teammate is random/bot
                        continue
                        
                    # Multi-policy dispatch
                    p_id = ("striker" if agent_id == 0 else "goalie") if TEAM_MODE else "default_policy"
                    try:
                        action[agent_id] = trainer.compute_single_action(agent_obs, policy_id=p_id, explore=False)
                    except:
                        # Fallback to default if policy names mismatch
                        action[agent_id] = trainer.compute_single_action(agent_obs, explore=False)

            obs, reward, dones, info = env.step(action)
            total_reward += sum(reward.values())
            done = dones.get("__all__", False)
            
        print(f"Episode {episode+1}: combined reward = {total_reward:.3f}")

    env.close()
    ray.shutdown()
