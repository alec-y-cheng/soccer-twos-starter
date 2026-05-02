# Privileged Information (Simple) Agent

**Agent name:** PPO_PrivInfo_Simple

## Description

An experiment in **Observation Space Compression**. Instead of using complex 336-channel Raycasts, this agent was given absolute "GPS" coordinates for itself and the ball. The goal was to determine if a low-dimensional state space leads to faster strategy convergence.

- **Training script:** `train_ray_selfplay.py` (with `OBS_TYPE="simple"`)
- **Opponent:** Self-Play
- **Observation Space:** **4-dimensional** (Relative `[ball_x, ball_y, goal_x, goal_y]`)
- **Reward:** Sparse goal signal + Distance-to-ball shaping.
- **Variation:** `team_vs_policy`
- **Action:** MultiDiscrete([3,3,3])

## Key Observations

- **Learning Speed:** This agent learned to navigate to the ball almost instantly (within < 1M timesteps) because it didn't have to learn how to interpret "vision" from raycasts.
- **Strategic Ceiling:** While fast to learn, this agent lacked the ability to "see" opponents or walls that weren't explicitly in its 4-dim vector, making it vulnerable to collisions and advanced blocking strategies.

## Checkpoint

`ray_results/PPO_priv_info_simple/`
