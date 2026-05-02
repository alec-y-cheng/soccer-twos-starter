# Goalie Self-Play Agent

**Agent name:** PPO_Goalie_SelfPlay

## Description

PPO agent trained for defensive specialization. This experiment introduced the first iteration of the `GoalieRewardWrapper`, heavily incentivizing ball interception and blocking within the defensive half.

- **Training script:** `train_ray_selfplay_goalie.py`
- **Opponent:** Self-Play
- **Reward:** Sparse goal signal + `GoalieRewardWrapper`:
  - `+1.0` (later bumped) for touching the ball inside the defensive zone.
  - `-5.0` penalty for conceding a goal.
  - Position-based rewards for staying between the ball and the net.
- **Variation:** `multiagent_player`
- **Observation:** Single-player raycasts (336-dim)
- **Action:** MultiDiscrete([3,3,3])

## Checkpoint

`ray_results/PPO_selfplay_goalie/`

## Batch script

`scripts/soccertwos_goalie_train.batch`
