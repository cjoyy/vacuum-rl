"""
Retrain v2 — standalone training script for DQN, A2C, SAC.

Saves models as {algorithm}_v3_policy_seed_{seed}.zip alongside existing v2 baselines.
Run locally or on Colab — NOT inside HF Spaces.

Usage:
    python ml/retrain_v2.py --algorithms dqn a2c sac --total_timesteps 200000 --seed 12345
    python ml/retrain_v2.py --algorithms dqn --dqn_lr 0.0005 --dqn_exploration_fraction 0.3
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
