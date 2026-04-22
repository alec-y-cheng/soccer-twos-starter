from random import uniform as randfloat
import math
import gym
import numpy as np
from ray.rllib import MultiAgentEnv
import soccer_twos


class SimpleObservationWrapper(gym.core.Wrapper):  
    def __init__(self, env):
        super().__init__(env)
        self.cached_info = None
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )
    
    def reset(self, **kwargs):
        self.cached_info = None
        obs = self.env.reset(**kwargs)
        return self.observation(obs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.cached_info = info
        return self.observation(obs), reward, done, info
        
    def observation(self, obs):
        if isinstance(obs, dict):
            return {pid: self.get_custom_obs(pid, ob) for pid, ob in obs.items()}
        # If it's single agent BUT it's a team... obs is shape (672,)
        if isinstance(obs, np.ndarray) and len(obs) == 672:
            return np.concatenate((self.get_custom_obs(obs[:336]), self.get_custom_obs(obs[336:])))
        return self.get_custom_obs(0, obs)
        
    def get_custom_obs(self, pid, obs):
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        goal_hits = obs[1::8]
        distances = obs[7::8]
        num_rays = len(goal_hits)
        
        if num_rays == 0:
            return np.zeros(4, dtype=np.float32)

        ball_x, ball_y, goal_x, goal_y = 0.0, 0.0, 0.0, 0.0

        for i in range(num_rays):
            angle = (i / num_rays) * 2 * np.pi
            dist = distances[i]
            if goal_hits[i] == 1.0:
                goal_x = dist * np.cos(angle)
                goal_y = dist * np.sin(angle)
                
        p_info = None if self.cached_info is None else (
            self.cached_info.get(pid, self.cached_info) if isinstance(self.cached_info, dict) else self.cached_info
        )
        
        if isinstance(p_info, dict) and "player_info" in p_info and "ball_info" in p_info:
            px, pz = p_info["player_info"]["position"]
            bx, bz = p_info["ball_info"]["position"]
            rot_y = p_info["player_info"]["rotation_y"]
            
            dx = bx - px
            dz = bz - pz
            
            theta = np.radians(rot_y)
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            
            rel_x = dx * cos_t + dz * sin_t
            rel_y = -dx * sin_t + dz * cos_t
            
            ball_x = rel_x / 20.0
            ball_y = rel_y / 20.0
                
        return np.array([ball_x, ball_y, goal_x, goal_y], dtype=np.float32)


class ExtendedObservationWrapper(gym.core.Wrapper):  
    def __init__(self, env):
        super().__init__(env)
        self.cached_info = None
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
        )
    
    def reset(self, **kwargs):
        self.cached_info = None
        obs = self.env.reset(**kwargs)
        return self.observation(obs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.cached_info = info
        return self.observation(obs), reward, done, info
        
    def observation(self, obs):
        if isinstance(obs, dict):
            return {pid: self.get_custom_obs(pid, ob) for pid, ob in obs.items()}
        return self.get_custom_obs(0, obs)
        
    def get_custom_obs(self, pid, obs):
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        goal_hits = obs[1::8]
        wall_hits = obs[3::8]
        distances = obs[7::8]
        num_rays = len(goal_hits)
        
        if num_rays == 0:
            return np.zeros(10, dtype=np.float32)

        ball_x, ball_y, ball_seen, goal_x, goal_y, goal_seen = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        for i in range(num_rays):
            angle = (i / num_rays) * 2 * np.pi
            dist = distances[i]
            if goal_hits[i] == 1.0:
                goal_x = dist * np.cos(angle)
                goal_y = dist * np.sin(angle)
                goal_seen = 1.0
                
        p_info = None if self.cached_info is None else (
            self.cached_info.get(pid, self.cached_info) if isinstance(self.cached_info, dict) else self.cached_info
        )
        
        if isinstance(p_info, dict) and "player_info" in p_info and "ball_info" in p_info:
            px, pz = p_info["player_info"]["position"]
            bx, bz = p_info["ball_info"]["position"]
            rot_y = p_info["player_info"]["rotation_y"]
            
            dx = bx - px
            dz = bz - pz
            
            theta = np.radians(rot_y)
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            
            rel_x = dx * cos_t + dz * sin_t
            rel_y = -dx * sin_t + dz * cos_t
            
            ball_x = rel_x / 20.0
            ball_y = rel_y / 20.0
            ball_seen = 1.0

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
        # Ball hits are at index 0 in the one-hot encoding
        ball_hits = player_obs[0::8]
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

class PrivilegedRewardShaping(gym.core.Wrapper): 
    # Exact same logic as CustomRewardShaping but guarantees the agent gets 
    # rewarded based on absolute Euclidean distance regardless of line-of-sight.
    def __init__(self, env):
        super().__init__(env)
        self.prev_dist = {}
    
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        is_multiagent = isinstance(obs, dict)

        if is_multiagent:
            self.prev_dist = {pid: 1.0 for pid in obs.keys()}
        else:
            self.prev_dist = 1.0
        return obs

    def step(self, action):
        import math
        obs, reward, done, info = self.env.step(action)
        is_multiagent = isinstance(obs, dict)
        
        if is_multiagent:
            shaped_reward = {}
            for pid, player_obs in obs.items():
                shaped_reward[pid] = reward[pid] + self._calculate_shaping(pid, info)
            return obs, shaped_reward, done, info
        else:
            shaped_reward = reward + self._calculate_shaping(0, info)
            return obs, shaped_reward, done, info

    def _calculate_shaping(self, pid, info):
        import math
        import numpy as np
        shaping = 0.0
        p_info = info.get(pid, info) if isinstance(info, dict) else {}
        
        if isinstance(p_info, dict) and "player_info" in p_info and "ball_info" in p_info:
            player_pos = np.array(p_info["player_info"]["position"])
            ball_pos = np.array(p_info["ball_info"]["position"])
            
            # 1. Proximity Reward (Distance to Ball)
            # Normalize absolute distance to 0-1 raycast scale (approx 20 units)
            closest_dist = min(1.0, math.dist(player_pos, ball_pos) / 20.0)
            
            if isinstance(self.prev_dist, dict):
                prev = self.prev_dist.get(pid, closest_dist)
                self.prev_dist[pid] = closest_dist
            else:
                prev = self.prev_dist
                self.prev_dist = closest_dist

            # Bumped from 0.005 to 0.02 to make the "pull" towards the ball stronger
            shaping += 0.02 * (prev - closest_dist) if closest_dist < 1.0 else 0.0

            # 2. Facing Reward: Point points towards the ball
            # player_info["rotation_y"] is in degrees. Convert to unit vector.
            # Unity 0 deg is Z forward, 90 is X right. Let's assume standard unit circle for sim.
            angle_rad = math.radians(p_info["player_info"]["rotation_y"])
            forward_vec = np.array([math.sin(angle_rad), math.cos(angle_rad)]) # Unity Z-forward mapping
            
            to_ball_vec = ball_pos - player_pos
            to_ball_norm = np.linalg.norm(to_ball_vec)
            if to_ball_norm > 0.1:
                to_ball_vec /= to_ball_norm
                dot = np.dot(forward_vec, to_ball_vec)
                # If facing the ball (dot > 0.5), give a small continuous bonus
                if dot > 0.5:
                    shaping += 0.001 * dot

        return shaping

class GoalieRewardWrapper(gym.core.Wrapper):
    """
    Rewards defensive positioning: staying close to own goal, 
    blocking the line of sight between the ball and the goal, 
    and intercepting the ball in the defensive half.
    """
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        is_multiagent = isinstance(obs, dict)
        
        if is_multiagent:
            shaped_reward = {}
            for pid, player_obs in obs.items():
                shaping = self._calculate_goalie_shaping(pid, info)
                # Massive penalty if a goal is conceded (-1 from Unity)
                if reward[pid] < -0.5:
                    shaping -= 5.0
                shaped_reward[pid] = reward[pid] + shaping
            return obs, shaped_reward, done, info
        else:
            shaping = self._calculate_goalie_shaping(0, info)
            if reward < -0.5:
                shaping -= 5.0
            shaped_reward = reward + shaping
            return obs, shaped_reward, done, info

    def _calculate_goalie_shaping(self, pid, info):
        shaping = 0.0
        
        p_info = info.get(pid, info) if isinstance(info, dict) else {}
        
        if isinstance(p_info, dict) and "player_info" in p_info and "ball_info" in p_info:
            player_pos = np.array(p_info["player_info"]["position"])
            ball_pos = np.array(p_info["ball_info"]["position"])
            
            # Determine which goal this agent is defending
            # PIDs 0, 1 defend left side (x=-14). PIDs 2, 3 defend right side (x=14).
            if pid in [0, 1]:
                goal_pos = np.array([-14.0, 0.0])
            else:
                goal_pos = np.array([14.0, 0.0])
                
            # 1. Leash: Penalize moving out of the box, reward staying in it
            dist_to_goal = math.dist(player_pos, goal_pos)
            if dist_to_goal > 8.0:
                shaping -= 0.01 * (dist_to_goal - 8.0)
            else:
                shaping += 0.005 

            # 2. Angling: Reward being on the line between the Ball and the Goal
            gb = ball_pos - goal_pos
            gb_len = np.linalg.norm(gb)
            
            if gb_len > 0.1:
                ga = player_pos - goal_pos
                t = np.dot(ga, gb) / (gb_len * gb_len)
                t = max(0.0, min(1.0, t)) # Clamp to line segment
                target_pos = goal_pos + t * gb
                
                dist_to_line = math.dist(player_pos, target_pos)
                if dist_to_line < 2.0:
                    shaping += 0.01 * (2.0 - dist_to_line)
                    
            # 3. Interception: Massive bonus for touching ball in defensive zone
            dist_to_ball = math.dist(player_pos, ball_pos)
            if dist_to_ball < 1.5 and dist_to_goal < 10.0:
                shaping += 5.0 # Increased from 1.0 to heavily incentivize blocking/intercepting
                
        return shaping

class TeamRewardWrapper(gym.core.Wrapper):
    """
    Applies Goalie rewards to Goalies (PIDs 1, 3) and Striker rewards to Strikers (PIDs 0, 2).
    """
    def __init__(self, env):
        super().__init__(env)
        self.striker_logic = PrivilegedRewardShaping(env)
        self.goalie_logic = GoalieRewardWrapper(env)
        
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        is_multiagent = isinstance(obs, dict)
        
        if is_multiagent:
            shaped_reward = {}
            for pid, player_obs in obs.items():
                if pid in [1, 3]: # Goalies
                    shaping = self.goalie_logic._calculate_goalie_shaping(pid, info)
                    if reward[pid] < -0.5: # Conceded Goal
                        shaping -= 5.0
                else: # Strikers (0, 2)
                    shaping = self.striker_logic._calculate_shaping(pid, info)
                    if reward[pid] > 0.5: # Scored Goal!
                        shaping += 10.0 # Huge bonus for actually hitting the net
                    
                shaped_reward[pid] = reward[pid] + shaping
            return obs, shaped_reward, done, info
        else:
            return obs, reward, done, info

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
        import os
        # Use SLURM_JOB_ID to guarantee a unique port block for each concurrent Slurm job.
        job_id = int(os.environ.get("SLURM_JOB_ID", "0"))
        if job_id == 0:
            import random
            job_id = random.randint(100, 999)
        # Incorporate job_id for uniqueness, but keep the final worker_id under 10000
        # to prevent ML-Agents from overflowing when it opens offset auxiliary ports.
        # Ensure a gap of at least 100 ports per worker to prevent "Address already in use" overlapping!
        env_config["worker_id"] = (
            job_id 
            + env_config.worker_index * 100
            + env_config.vector_index
        ) % 10000
    env = soccer_twos.make(**env_config)
    
    # Toggle Rewards here:
    reward_type = env_config.get("reward_shaping", "custom")
    if reward_type == "custom":
        env = CustomRewardShaping(env)
    elif reward_type == "privileged":
        env = PrivilegedRewardShaping(env)
    elif reward_type == "goalie":
        env = GoalieRewardWrapper(env)
    elif reward_type == "team":
        env = TeamRewardWrapper(env)
    elif reward_type == "dummy":
        env = BallTouchReward(env)
    # else: no reward shaping (standard unity rewards)
    
    """
    obs_type = env_config.get("obs_type", "simple")
    if obs_type == "simple":
        env = SimpleObservationWrapper(env)
    elif obs_type == "extended":
        env = ExtendedObservationWrapper(env)
    """

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
