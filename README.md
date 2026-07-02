# vacuum-rl

Web app monorepo for running and visualizing trained reinforcement-learning policies on the Vacuum Cleaner Gymnasium environment.

## Current Stage

Stage 1 sets up the `ml/` package with:

- `ml/env.py`: Gymnasium `VacuumCleaningEnv` copied from the reference project.
- `ml/config/config_1.yaml`: 2x2 environment config.
- `ml/config/config_2.yaml`: 7x7 environment config used by the trained policies.
- `ml/models/`: seed `12345` checkpoints for DQN, PPO, TRPO, A2C, and SAC.
- `ml/requirements.txt`: Python dependencies for environment loading and policy inference.

## Model Loading Notes

DQN, PPO, A2C, and SAC checkpoints are Stable-Baselines3 models. TRPO is provided by `sb3-contrib`.

SAC was trained through a continuous-action wrapper that maps a one-dimensional Box action back to the discrete environment action set.
