---
title: Vacuum RL
emoji: 🧹
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# vacuum-rl

Web app monorepo for running and visualizing trained reinforcement-learning policies on the Vacuum Cleaner Gymnasium environment.

## Current Stage

Stage 1 sets up the `ml/` package with:

- `ml/env.py`: Gymnasium `VacuumCleaningEnv` copied from the reference project.
- `ml/config/config_1.yaml`: 2x2 environment config.
- `ml/config/config_2.yaml`: 7x7 environment config used by the trained policies.
- `ml/models/`: seed `12345` checkpoints for DQN, PPO, TRPO, A2C, and SAC.
- `ml/requirements.txt`: Python dependencies for environment loading and policy inference.

Stage 2 adds a FastAPI backend:

- `GET /algorithms`
- `POST /reset`
- `WebSocket /ws/step`
- startup policy cache for SB3 and sb3-contrib models
- Dockerfile for Hugging Face Spaces on CPU

Stage 3 adds a React + Vite + Tailwind frontend:

- canvas grid renderer
- algorithm/action controls
- reset/play/pause/step/speed controls
- WebSocket state updates
- `VITE_API_URL` for backend URL configuration

## Model Loading Notes

DQN, PPO, A2C, and SAC checkpoints are Stable-Baselines3 models. TRPO is provided by `sb3-contrib`.

SAC was trained through a continuous-action wrapper that maps a one-dimensional Box action back to the discrete environment action set.

## Local Development

Backend:

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```
