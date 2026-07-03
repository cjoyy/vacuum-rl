---
title: Vacuum RL
emoji: 🧹
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# vacuum-rl

![Lint](https://github.com/cjoyy/vacuum-rl/actions/workflows/lint.yml/badge.svg)

Web app monorepo for running and visualizing trained reinforcement-learning policies on the Vacuum Cleaner Gymnasium environment. Features a single-grid interactive demo and a multi-grid arena for side-by-side algorithm comparison.

## Screenshots

> _Coming soon: demo single-grid screenshot & arena mode screenshot/GIF._

### Single-Grid Demo

Live WebSocket-powered demo with canvas grid rendering, algorithm selection (DQN, PPO, TRPO, A2C, SAC), manual/auto step, speed control, battery meter, and episode stats.

### Arena Mode

Multi-grid comparison at `/arena`. Select up to 4 algorithms and run them simultaneously on identical grid configurations (same seed, same dirt layout) for fair visual comparison.

## Features

- **Single-Grid Demo** — Observe policy decisions step by step with real-time WebSocket updates.
- **Multi-Grid Arena** — Compare up to 4 algorithms running lockstep on identical environments.
- **5 RL Algorithms** — DQN, PPO, TRPO, A2C, SAC (Stable-Baselines3 & sb3-contrib).
- **i18n** — English (default) and Indonesian language support with EN/ID switcher.
- **Canvas Renderer** — Custom 2D canvas grid with dirt levels, obstacles, battery status, and robot animation.

## Limitations

Per the project paper (Table 4 policy-quality results), not all algorithms achieve the same level of performance:

| Algorithm | Status |
|-----------|--------|
| **PPO** | Best performing — 71.67% success rate |
| **TRPO** | Competitive with PPO |
| **DQN** | Stuck in conservative local optimum — rarely completes full grid cleanup |
| **A2C** | Similar issue to DQN — insufficient exploration |
| **SAC** | Continuous-to-discrete mapping adds instability; also struggles |

See `docs/hyperparameter-improvements.md` for analysis and suggested retraining hyperparameters.

## Model Loading Notes

DQN, PPO, A2C, and SAC checkpoints are Stable-Baselines3 models. TRPO is provided by `sb3-contrib`.

SAC was trained through a continuous-action wrapper that maps a one-dimensional Box action back to the discrete environment action set.

All models are loaded and cached at startup (not lazy-load per request). The `/health` endpoint reports per-model load times.

## Retraining

If you want to experiment with improved hyperparameters for DQN, A2C, or SAC:

```bash
cd ml
pip install -r requirements.txt
python retrain_v2.py --algorithms dqn a2c sac --total_timesteps 300000
```

This produces `{algo}_v3_policy_seed_12345.zip` files alongside the existing v2 baselines — no models are replaced automatically. See `docs/hyperparameter-improvements.md` for the full rationale.

## Local Development

### Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

### Environment Variables

Copy `.env.example` to `.env` and adjust as needed:

```bash
cp .env.example .env
```

| Variable | Default | Description |
|----------|---------|-------------|
| `CORS_ORIGINS` | `http://localhost:5173` | Allowed CORS origins (comma-separated) |
| `VITE_API_URL` | `http://localhost:8000` | Backend API URL for frontend dev server |

## Production Docker

```bash
docker build -t vacuum-rl .
docker run --rm -p 7860:7860 vacuum-rl
```
