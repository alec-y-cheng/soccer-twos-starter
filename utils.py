from random import uniform as randfloat

import gym
import numpy as np
from ray.rllib import MultiAgentEnv
import soccer_twos

class CustomRewardShaping(gym.core.Wrapper): # distance to ball metric
    def __init__(self, env):
        super().__init__(env)
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        
        # Handle both multi-agent (dict) and single-agent (ndarray/float) cases
        is_multiagent = isinstance(obs, dict)
        
        if is_multiagent:
            shaped_reward = {}
            for player_id, player_obs in obs.items():
                shaped_reward[player_id] = reward[player_id] + self._calculate_shaping(player_obs)
            return obs, shaped_reward, done, info
        else:
            # Single agent case
            shaped_reward = reward + self._calculate_shaping(obs)
            return obs, shaped_reward, done, info

    def _calculate_shaping(self, player_obs):
        # extract every 8th element (the boolean hits for the ball)
        # and every 8th element starting at index 7 (the distances)
        ball_hits = player_obs[0::8]
        distances = player_obs[7::8]
        
        closest_dist = 1.0
        # find closest hit
        for i in range(len(ball_hits)):
            if ball_hits[i] == 1.0:
                if distances[i] < closest_dist:
                    closest_dist = distances[i]
        
        if closest_dist < 1.0:
            return 0.005 * (1.0 - closest_dist)
        return 0.0

class RLLibWrapper(gym.core.Wrapper, MultiAgentEnv):
    """
    A RLLib wrapper so our env can inherit from MultiAgentEnv.
    """

    pass


def create_rllib_env(env_config: dict = {}):
    """
    Creates a RLLib environment and prepares it to be instantiated by Ray workers.
    Args:
        env_config: configuration for the environment.
            You may specify the following keys:
            - variation: one of soccer_twos.EnvType. Defaults to EnvType.multiagent_player.
            - opponent_policy: a Callable for your agent to train against. Defaults to a random policy.
    """
    if hasattr(env_config, "worker_index"):
        env_config["worker_id"] = (
            env_config.worker_index * env_config.get("num_envs_per_worker", 1)
            + env_config.vector_index
        )
    env = soccer_twos.make(**env_config)
    # env = TransitionRecorderWrapper(env)
    env = CustomRewardShaping(env)
    if "multiagent" in env_config and not env_config["multiagent"]:
        # is multiagent by default, is only disabled if explicitly set to False
        return env
    return RLLibWrapper(env)


def sample_vec(range_dict):
    return [
        randfloat(range_dict["x"][0], range_dict["x"][1]),
        randfloat(range_dict["y"][0], range_dict["y"][1]),
    ]


def sample_val(range_tpl):
    return randfloat(range_tpl[0], range_tpl[1])


def sample_pos_vel(range_dict):
    _s = {}
    if "position" in range_dict:
        _s["position"] = sample_vec(range_dict["position"])
    if "velocity" in range_dict:
        _s["velocity"] = sample_vec(range_dict["velocity"])
    return _s


def sample_player(range_dict):
    _s = sample_pos_vel(range_dict)
    if "rotation_y" in range_dict:
        _s["rotation_y"] = sample_val(range_dict["rotation_y"])
    return _s
