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

# ── BASELINE: If True, Orange team is driven by Partner's Agent
PLAY_AGAINST_BASELINE = True
#BASELINE_PATH = 'ceia_baseline_agent/ray_results/PPO_selfplay_twos/PPO_Soccer_f475e_00000_0_2021-09-19_15-54-02/checkpoint_002449/checkpoint-2449-patched'
BASELINE_PATH = 'checkpoints/HU_PPO3_selfplay_agent/ray_results/PPO_selfplay/checkpoint-7500'

from utils import create_rllib_env

def get_config(observation_space, action_space, model_hiddens=[256, 256]):
    import gym
    config = {
        "num_gpus": 0, "num_workers": 0, "framework": "torch", "explore": False,
        "env": "DummyEnv", "disable_env_checking": True,
        "observation_space": observation_space, "action_space": action_space,
        "model": {"vf_share_layers": True, "fcnet_hiddens": model_hiddens, "fcnet_activation": "relu"},
    }
    
    if TEAM_MODE:
        config["multiagent"] = {
            "policies": {
                "striker": (None, observation_space, action_space, {}),
                "goalie": (None, observation_space, action_space, {}),
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
        def __init__(self, config):
            super().__init__()
            # Use the space provided in the trainer config if available, fallback to global
            self.observation_space = config.get("observation_space", observation_space)
            self.action_space = config.get("action_space", action_space)
        def reset(self): return {}
        def step(self, action): return {}, {}, {}, {}

    tune.registry.register_env("DummyEnv", lambda config: MAEnv(config)) 
    tune.registry.register_env("Soccer", create_rllib_env)

    # 1. Initialize your trainer (256 neurons)
    my_config = get_config(observation_space, action_space)
    my_config["env_config"] = {
        "observation_space": observation_space,
        "action_space": action_space,
    }
    trainer = PPOTrainer(config=my_config)
    trainer.restore(CHECKPOINT_PATH)
    print(f"✓ Loaded your checkpoint: {CHECKPOINT_PATH}")

    # 2. Partner setup (512 neurons)
    baseline_trainer = None
    if PLAY_AGAINST_BASELINE:
        # Save global state, force single-agent for baseline
        _old_team_mode = TEAM_MODE
        globals()['TEAM_MODE'] = False 
        
        # Partner uses Team-Agent architecture: 672 inputs (dual obs) and 18 outputs (dual joy)
        partner_obs_space = gym.spaces.Box(low=-1.0, high=2.0, shape=(672,), dtype=np.float32)
        partner_act_space = gym.spaces.MultiDiscrete([3, 3, 3, 3, 3, 3])
        
        b_config = get_config(partner_obs_space, partner_act_space, model_hiddens=[512, 512])
        
        # Ensure DummyEnv sees these spaces in its own 'config'
        b_config["env_config"] = {
            "observation_space": partner_obs_space,
            "action_space": partner_act_space,
        }
        
        # Restoration
        baseline_trainer = PPOTrainer(config=b_config)
        baseline_trainer.restore(BASELINE_PATH)
        print(f"✓ Loaded partner TEAM agent as baseline: {BASELINE_PATH}")
        
        # Restore global state
        globals()['TEAM_MODE'] = _old_team_mode

    # 3. Environment
    eval_env_config = {
        "variation": EnvType.multiagent_player, # Use player mode to get 4 distinct IDs
        "multiagent": False, 
        "flatten_branched": FLATTENED,
        "single_player": False,
        "obs_type": OBS_TYPE,
        "watch": True, "worker_id": 1,
    }
    env = create_rllib_env(eval_env_config)

    for episode in range(5):
        obs = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action = {}
            # --- ORANGE TEAM (2, 3) --- 
            if PLAY_AGAINST_BASELINE and baseline_trainer:
                # Concatenate Agent 2 and 3 observations for your partner's team-brain
                fused_orange_obs = np.concatenate([obs[2], obs[3]])
                fused_act = baseline_trainer.compute_single_action(fused_orange_obs, explore=False)
                # Split action back into two individual parts
                action[2] = fused_act[:3]
                action[3] = fused_act[3:]
            else:
                action[2] = env.action_space.sample()
                action[3] = env.action_space.sample()

            # --- BLUE TEAM (0, 1) ---
            for agent_id in [0, 1]:
                if agent_id == 1 and not CONTROL_BOTH_PLAYERS:
                    action[1] = env.action_space.sample()
                    continue
                    
                p_id = ("striker" if agent_id == 0 else "goalie") if TEAM_MODE else "default_policy"
                try:
                    action[agent_id] = trainer.compute_single_action(obs[agent_id], policy_id=p_id, explore=False)
                except:
                    action[agent_id] = trainer.compute_single_action(obs[agent_id], explore=False)

            obs, reward, dones, info = env.step(action)
            total_reward += sum(reward.values())
            done = dones.get("__all__", False)
            
        print(f"Episode {episode+1}: combined reward = {total_reward:.3f}")

    env.close()
    ray.shutdown()
