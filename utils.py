from random import uniform as randfloat

import gym
import numpy as np
from ray.rllib import MultiAgentEnv
import soccer_twos


class CustomObservationWrapper(gym.ObservationWrapper):  
    # logic for x and y relative position (compress obs space)
    # also implement logic for getting unstuck when things are not visible
    def __init__(self, env, obs_type="simple"):
        super().__init__(env)
        self.obs_type = obs_type

        if obs_type == "simple":
            shape = (4,)
        elif obs_type == "extended":
            shape = (10,)
        else:
            raise ValueError(f"Unknown obs_type: {obs_type}")

        self.observation_space = gym.spaces.Box(  # define observaiton shape
            low=-1.0, high=2.0, shape=shape, dtype=np.float32
        )
    
    def observation(self, obs):
        if isinstance(obs, dict):  #multi agent
            return {pid: self.get_custom_obs(ob) for pid, ob in obs.items()}
        return self.get_custom_obs(obs)  #single agent
        
    def get_custom_obs(self, obs):  # to ball and goal
        # convert polar to cartesian with x = r * cos(theta) and y = r * sin(theta), then invert the x and y for relative position
        wall_hits = obs[0::8]
        goal_hits = obs[1::8]
        ball_hits = obs[2::8]
        distances = obs[7::8]
        num_rays = len(ball_hits)

        ball_x, ball_y, ball_seen, goal_x, goal_y, goal_seen = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        for i in range(num_rays):
            angle = (i / num_rays) * 2 * np.pi
            dist = distances[i]

            if ball_hits[i] == 1.0:
                ball_x = dist * np.cos(angle)
                ball_y = dist * np.sin(angle)
                ball_seen = 1.0
            
            if goal_hits[i] == 1.0:
                goal_x = dist * np.cos(angle)
                goal_y = dist * np.sin(angle)
                goal_seen = 1.0
                
        if self.obs_type == "simple":
            return np.array([ball_x, ball_y, goal_x, goal_y], dtype=np.float32)

        # wall seen from 4 directions
        wall_f = distances[21] if wall_hits[21] else 1.0
        wall_b = distances[0]  if wall_hits[0]  else 1.0
        wall_l = distances[10] if wall_hits[10] else 1.0
        wall_r = distances[31] if wall_hits[31] else 1.0

        return np.array([ball_x, ball_y, ball_seen, goal_x, goal_y, goal_seen, wall_f, wall_b, wall_l, wall_r], dtype=np.float32)
    

class CustomRewardShaping(gym.core.Wrapper): # distance to ball metric
    def __init__(self, env):
        super().__init__(env)
        self.prev_dist = {}
    
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        is_multiagent = isinstance(obs, dict)

        if is_multiagent:
            self.prev_dist = {
                player_id: self._get_distance(player_obs)
                for player_id, player_obs in obs.items()
            }
        else:
            self.prev_dist = self._get_distance(obs)
    
        return obs

    def _get_distance(self, player_obs):
        # Ball hits are at index 2 in the one-hot encoding
        ball_hits = player_obs[2::8]
        distances = player_obs[7::8]

        closest_dist = 1.0
        for i in range(len(ball_hits)):
            if ball_hits[i] == 1.0:
                if distances[i] < closest_dist:
                    closest_dist = distances[i]

        return closest_dist
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        
        # Handle both multi-agent (dict) and single-agent (ndarray/float) cases
        is_multiagent = isinstance(obs, dict)
        
        if is_multiagent:
            shaped_reward = {}
            for player_id, player_obs in obs.items():
                shaped_reward[player_id] = reward[player_id] + self._calculate_shaping(player_id, player_obs)
            return obs, shaped_reward, done, info
        else:
            # Single agent case
            shaped_reward = reward + self._calculate_shaping(None, obs)
            return obs, shaped_reward, done, info

    def _calculate_shaping(self, pid, player_obs):
        closest_dist = self._get_distance(player_obs)
        
        if isinstance(self.prev_dist, dict):
            prev = self.prev_dist.get(pid, closest_dist)
            self.prev_dist[pid] = closest_dist
        else:
            prev = self.prev_dist
            self.prev_dist = closest_dist

        shaping = 0.005 * (prev - closest_dist) if closest_dist < 1.0 else 0.0

        return shaping

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
    
    obs_type = env_config.get("obs_type", "simple")
    if obs_type != "raw":
        env = CustomObservationWrapper(env, obs_type=obs_type)

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
