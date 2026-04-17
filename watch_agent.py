import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from soccer_twos import EnvType

# ── UPDATE THIS to point at your local checkpoint file ─────────────────────
CHECKPOINT_PATH = 'checkpoints/checkpoint_000975/checkpoint-975-patched'

TRAIN_CONFIG = {
    "num_gpus": 0,       
    "num_workers": 0,    
    "framework": "torch",
    "env": "Soccer",
    "env_config": {
        "variation": EnvType.team_vs_policy,
        "multiagent": False,
        "flatten_branched": True,
        "single_player": True,
        "opponent_policy": lambda *_: 0,
    },
    "model": {
        "vf_share_layers": True,
        "fcnet_hiddens": [256, 256],
        "fcnet_activation": "relu",
    },
    "explore": False,     
}

from utils import create_rllib_env

if __name__ == "__main__":
    ray.init(include_dashboard=False)
    tune.registry.register_env("Soccer", create_rllib_env)

    trainer = PPOTrainer(config=TRAIN_CONFIG)
    trainer.restore(CHECKPOINT_PATH)
    print(f"✓ Loaded checkpoint: {CHECKPOINT_PATH}")

    env = create_rllib_env({
        **TRAIN_CONFIG["env_config"],
        "watch": True,          # <── this opens the visual display AND slows it down to human speed
        "worker_id": 1,         # <── avoid port clash with old crashed windows!
    })

    for episode in range(5):
        obs = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        while not done:
            action = trainer.compute_single_action(obs, explore=False)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
        print(f"Episode {episode+1}: {steps} steps, reward = {total_reward:.3f}")

    env.close()
    ray.shutdown()
