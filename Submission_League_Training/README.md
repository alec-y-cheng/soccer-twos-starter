# League Training Phase 1

**Agent name:** PPO_League_Phase1

## Description

The first "Mixed-Opponent" experiment. Instead of pure self-play, this agent trained against a rotating roster to maximize exposure to different strategies and stabilize the learning process.

- **Training script:** `train_ray_league.py`
- **Opponent Mixture:**
  - 40% Self-Play (Current weights)
  - 40% CEIA Baseline (Hardcoded benchmark)
  - 20% Random Dummy (For robustness)
- **Reward:** `TeamRewardWrapper`
  - Routes Striker logic to Strikers and Goalie logic to Goalies.
  - Interception bonus increased to **+5.0**.
- **Variation:** `multiagent_player`
- **Observation:** Single-player raycasts (336-dim)
- **Action:** MultiDiscrete([3,3,3])

## Checkpoint

`ray_results/PPO_league_training/`

## Batch script

`scripts/soccertwos_team_train.batch`
