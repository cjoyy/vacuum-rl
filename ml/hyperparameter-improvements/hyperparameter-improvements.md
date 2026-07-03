# Hyperparameter Improvements for DQN, A2C, SAC

## Problem

Berdasarkan Table 4 paper, tiga algoritma — **DQN**, **A2C**, dan **SAC** — gagal mencapai performa yang memuaskan di environment VacuumCleaner-v2 (`config_2.yaml`). Ketiganya stuck di **local optimum konservatif**: agent belajar untuk tidak banyak bergerak, cenderung diam di dekat dock, dan tidak pernah menyelesaikan pembersihan seluruh grid.

Sebaliknya, **PPO** dan **TRPO** berhasil mencapai coverage dan episode return yang jauh lebih tinggi.

## Root Cause Analysis

### 1. Reset penalty terlalu dominan

```
reset_penalty: 600.0
```

Agent menerima penalti -600 saat battery habis dan di-reset ke dock. Ini adalah sinyal negatif terbesar dalam reward function. Akibatnya, agent belajar strategi "menghindari reset dengan cara tidak pernah jauh dari dock" — yang secara lokal optimal tapi secara global suboptimal.

### 2. Eksplorasi tidak memadai

| Algoritma | Parameter | Default SB3 | Dampak |
|-----------|-----------|-------------|--------|
| DQN | `exploration_fraction` | 0.1 (10% dari total training) | Terlalu cepat beralih ke eksploitasi sebelum sempat menemukan reward structure |
| DQN | `exploration_final_eps` | 0.05 | Hampir no noise setelah fase eksplorasi |
| A2C | `ent_coef` | 0.0 | Tanpa entropy bonus, policy collapse ke deterministik |
| SAC | (default) | `ent_coef="auto"` | Seringkali alpha mengecil terlalu cepat |

### 3. Horizon terlalu pendek (A2C)

A2C secara default menggunakan `n_steps=5`. Dengan grid 7x7 dan kebutuhan untuk membersihkan + kembali ke dock, satu episode membutuhkan puluhan langkah. Nilai 5 tidak cukup bagi advantage estimation untuk melihat konsekuensi jangka panjang dari suatu action.

### 4. Learning rate kurang stabil

DQN dan SAC menggunakan learning rate yang relatif tinggi untuk kompleksitas reward function ini, menyebabkan Q-value over-estimation dan konvergensi ke policy yang salah.

## Recommended Hyperparameter Changes

### DQN

| Parameter | Default SB3 | Recommended | Alasan |
|-----------|-------------|-------------|--------|
| `exploration_fraction` | 0.1 | **0.3** | 3x lebih lama eksplorasi agar agent bisa menemukan cleaning reward sebelum konvergen |
| `exploration_final_eps` | 0.05 | **0.15** | Random noise residual membantu agent keluar dari local optimum |
| `learning_rate` | 1e-4 | **5e-5** | Lebih stabil, mencegah Q-value oscillation |
| `batch_size` | 32 | **64** | Sample lebih banyak per update, estimasi lebih akurat |
| `buffer_size` | 1,000,000 | **500,000** | Buffer lebih kecil = replay experience lebih segar |

### A2C

| Parameter | Default SB3 | Recommended | Alasan |
|-----------|-------------|-------------|--------|
| `n_steps` | 5 | **50** | Horizon 10x lebih panjang agar advantage estimation bisa menangkap delayed reward |
| `ent_coef` | 0.0 | **0.01** | Entropy bonus mendorong eksplorasi |
| `learning_rate` | 7e-4 | **3e-4** | Update lebih konservatif |
| `normalize_advantage` | False | **True** | Stabilisasi training dengan rollout lebih panjang |
| `gae_lambda` | 1.0 | **0.95** | Bias-variance tradeoff lebih baik untuk n_steps besar |

### SAC

| Parameter | Default SB3 | Recommended | Alasan |
|-----------|-------------|-------------|--------|
| `learning_rate` | 3e-4 | **1e-4** | Update Q-function lebih stabil di reward landscape kompleks |
| `batch_size` | 256 | **512** | Gradient lebih akurat dengan sample lebih banyak |
| `tau` | 0.005 | **0.01** | Target network update 2x lebih cepat, propagating sparse cleaning reward |
| `buffer_size` | 1,000,000 | **500,000** | Sama dengan DQN — buffer lebih segar |

## Training Script

Berikut adalah script `ml/retrain_v2.py` untuk menjalankan retraining.

### Cara Penggunaan

```bash
# Retrain default (DQN, A2C, SAC) dengan hyperparameter baru
python ml/retrain_v2.py

# Retrain dengan total timesteps dan override spesifik
python ml/retrain_v2.py --algorithms dqn a2c sac --total_timesteps 500000

# Override per-algorithm
python ml/retrain_v2.py --algorithms dqn --dqn_lr 0.0001 --dqn_exploration_fraction 0.4

# Retrain SAC saja dengan batch size lebih besar
python ml/retrain_v2.py --algorithms sac --sac_batch_size 1024

# Skip eval callback (lebih cepat)
python ml/retrain_v2.py --no_eval
```

### Script

```python
"""
Retrain v2 — standalone training script for DQN, A2C, SAC.

Saves models as {algorithm}_v3_policy_seed_{seed}.zip alongside existing v2 baselines.
Run locally or on Colab — NOT inside HF Spaces.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from env import VacuumCleaningEnv
from stable_baselines3 import A2C, DQN, SAC
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("retrain_v2")

SEED = 12345
TOTAL_TIMESTEPS = 300_000
EVAL_FREQ = 10_000
EVAL_EPISODES = 10
MODELS_DIR = Path(__file__).resolve().parent / "models"

DEFAULT_HYPERPARAMS = {
    "dqn": {
        "learning_rate": 5e-5,
        "buffer_size": 500_000,
        "batch_size": 64,
        "gamma": 0.99,
        "tau": 1.0,
        "target_update_interval": 1000,
        "train_freq": 4,
        "gradient_steps": 1,
        "exploration_fraction": 0.3,
        "exploration_final_eps": 0.15,
        "exploration_initial_eps": 1.0,
        "max_grad_norm": 10,
        "policy": "MlpPolicy",
    },
    "a2c": {
        "learning_rate": 3e-4,
        "n_steps": 50,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "use_rms_prop": True,
        "normalize_advantage": True,
        "policy": "MlpPolicy",
    },
    "sac": {
        "learning_rate": 1e-4,
        "buffer_size": 500_000,
        "batch_size": 512,
        "tau": 0.01,
        "gamma": 0.99,
        "train_freq": 1,
        "gradient_steps": 1,
        "ent_coef": "auto",
        "target_update_interval": 1,
        "policy": "MlpPolicy",
    },
}


def make_env(config_name: str = "config_2.yaml", seed: int = SEED) -> VacuumCleaningEnv:
    return VacuumCleaningEnv(config_name=config_name)


def train_algorithm(
    algorithm: str,
    total_timesteps: int,
    hyperparams: dict,
    seed: int,
    eval_callback: bool = True,
) -> str:
    algo_dir = MODELS_DIR / algorithm
    algo_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(seed=seed)
    env = Monitor(env)

    algo_class = {"dqn": DQN, "a2c": A2C, "sac": SAC}[algorithm]
    logger.info("Initializing %s with hyperparams: %s", algorithm.upper(), hyperparams)
    model = algo_class(env=env, seed=seed, verbose=0, **hyperparams)

    callbacks = []
    if eval_callback:
        eval_env = Monitor(make_env(seed=seed + 1))
        callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=500, verbose=1)
        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=str(algo_dir / f"{algorithm}_v3_best"),
            log_path=str(algo_dir),
            eval_freq=EVAL_FREQ,
            n_eval_episodes=EVAL_EPISODES,
            deterministic=True,
            callback_on_new_best=callback_on_best,
        )
        callbacks.append(eval_cb)

    logger.info("Starting training %s for %d timesteps...", algorithm.upper(), total_timesteps)
    start = time.perf_counter()
    model.learn(total_timesteps=total_timesteps, callback=callbacks)
    elapsed = time.perf_counter() - start
    logger.info(
        "%s training done in %.1fs (%.0f steps/s)",
        algorithm.upper(), elapsed, total_timesteps / elapsed,
    )

    save_path = str(algo_dir / f"{algorithm}_v3_policy_seed_{seed}.zip")
    model.save(save_path)
    logger.info("Model saved to %s", save_path)

    env.close()
    if eval_callback:
        eval_env.close()

    return save_path


def main():
    parser = argparse.ArgumentParser(description="Retrain RL algorithms for vacuum-cleaner env")
    parser.add_argument(
        "--algorithms", nargs="+", choices=["dqn", "a2c", "sac"],
        default=["dqn", "a2c", "sac"],
        help="Algorithms to retrain (default: dqn a2c sac)",
    )
    parser.add_argument(
        "--total_timesteps", type=int, default=TOTAL_TIMESTEPS,
        help="Total training steps per algorithm",
    )
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    parser.add_argument("--no_eval", action="store_true", help="Skip eval callback")
    parser.add_argument("--dqn_lr", type=float, default=None, help="Override DQN learning_rate")
    parser.add_argument(
        "--dqn_exploration_fraction", type=float, default=None,
        help="Override DQN exploration_fraction",
    )
    parser.add_argument("--a2c_lr", type=float, default=None, help="Override A2C learning_rate")
    parser.add_argument("--a2c_n_steps", type=int, default=None, help="Override A2C n_steps")
    parser.add_argument("--sac_lr", type=float, default=None, help="Override SAC learning_rate")
    parser.add_argument("--sac_batch_size", type=int, default=None, help="Override SAC batch_size")

    args = parser.parse_args()

    for algorithm in args.algorithms:
        hp = dict(DEFAULT_HYPERPARAMS[algorithm])
        overrides = {
            "dqn": {"learning_rate": args.dqn_lr, "exploration_fraction": args.dqn_exploration_fraction},
            "a2c": {"learning_rate": args.a2c_lr, "n_steps": args.a2c_n_steps},
            "sac": {"learning_rate": args.sac_lr, "batch_size": args.sac_batch_size},
        }[algorithm]
        for key, value in overrides.items():
            if value is not None:
                hp[key] = value

        save_path = train_algorithm(
            algorithm=algorithm,
            total_timesteps=args.total_timesteps,
            hyperparams=hp,
            seed=args.seed,
            eval_callback=not args.no_eval,
        )
        logger.info("=== %s retrained → %s ===", algorithm.upper(), save_path)


if __name__ == "__main__":
    main()
```
