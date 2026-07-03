from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("vacuum-rl.simulation")

ROOT_DIR = Path(__file__).resolve().parents[2]
ML_DIR = ROOT_DIR / "ml"
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from env import VacuumCleaningEnv  # noqa: E402


@dataclass(frozen=True)
class AlgorithmSpec:
    id: str
    name: str
    source: str
    model_path: Path
    continuous_policy: bool = False


ALGORITHMS: dict[str, AlgorithmSpec] = {
    "dqn": AlgorithmSpec(
        id="dqn",
        name="DQN",
        source="stable_baselines3",
        model_path=ML_DIR / "models" / "dqn" / "dqn_v2_policy_seed_12345.zip",
    ),
    "ppo": AlgorithmSpec(
        id="ppo",
        name="PPO",
        source="stable_baselines3",
        model_path=ML_DIR / "models" / "ppo" / "ppo_v2_policy_seed_12345.zip",
    ),
    "trpo": AlgorithmSpec(
        id="trpo",
        name="TRPO",
        source="sb3_contrib",
        model_path=ML_DIR / "models" / "trpo" / "trpo_v2_policy_seed_12345.zip",
    ),
    "a2c": AlgorithmSpec(
        id="a2c",
        name="A2C",
        source="stable_baselines3",
        model_path=ML_DIR / "models" / "a2c" / "a2c_v2_policy_seed_12345.zip",
    ),
    "sac": AlgorithmSpec(
        id="sac",
        name="SAC",
        source="stable_baselines3",
        model_path=ML_DIR / "models" / "sac" / "sac_v2_policy_seed_12345.zip",
        continuous_policy=True,
    ),
}


class PolicyCache:
    def __init__(self) -> None:
        self._models: dict[str, Any] = {}
        self._load_errors: dict[str, str] = {}
        self._load_times: dict[str, float] = {}
        self._startup_duration: float = 0.0

    @property
    def load_errors(self) -> dict[str, str]:
        return dict(self._load_errors)

    @property
    def load_times(self) -> dict[str, float]:
        return dict(self._load_times)

    @property
    def started(self) -> bool:
        return bool(self._load_times) or bool(self._load_errors)

    def load_all(self) -> None:
        start = time.perf_counter()
        for algorithm in ALGORITHMS:
            algo_start = time.perf_counter()
            try:
                self.get(algorithm)
                elapsed = time.perf_counter() - algo_start
                self._load_times[algorithm] = elapsed
                logger.info("Loaded %s in %.2fs", ALGORITHMS[algorithm].name, elapsed)
            except RuntimeError:
                elapsed = time.perf_counter() - algo_start
                self._load_times[algorithm] = elapsed
                continue
        self._startup_duration = time.perf_counter() - start
        loaded = len(self._models)
        failed = len(self._load_errors)
        logger.info(
            "Startup model loading complete: %d loaded, %d failed in %.2fs",
            loaded, failed, self._startup_duration,
        )

    def get(self, algorithm: str) -> Any:
        algorithm = normalize_algorithm(algorithm)
        if algorithm in self._models:
            return self._models[algorithm]
        if algorithm in self._load_errors:
            raise RuntimeError(self._load_errors[algorithm])

        spec = ALGORITHMS[algorithm]
        try:
            if spec.source == "sb3_contrib":
                from sb3_contrib import TRPO

                model_class = TRPO
            else:
                from stable_baselines3 import A2C, DQN, PPO, SAC

                model_class = {
                    "a2c": A2C,
                    "dqn": DQN,
                    "ppo": PPO,
                    "sac": SAC,
                }[algorithm]
            model = model_class.load(str(spec.model_path), device="cpu")
        except Exception as exc:  # keep API alive when local deps/models are absent
            message = f"Failed to load {spec.name} policy from {spec.model_path}: {exc}"
            self._load_errors[algorithm] = message
            raise RuntimeError(message) from exc

        self._models[algorithm] = model
        return model


policy_cache = PolicyCache()

ARENA_MAX_INSTANCES = 4


def make_arena_envs(algorithms: list[str], seed: int = 12345) -> dict[str, VacuumCleaningEnv]:
    envs: dict[str, VacuumCleaningEnv] = {}
    for algorithm in algorithms[:ARENA_MAX_INSTANCES]:
        algorithm = normalize_algorithm(algorithm)
        env = make_env(seed=seed)
        envs[algorithm] = env
    return envs


def arena_reset(envs: dict[str, VacuumCleaningEnv], seed: int = 12345) -> list[dict[str, Any]]:
    states: list[dict[str, Any]] = []
    for algorithm, env in envs.items():
        states.append(reset_env(env, algorithm=algorithm, seed=seed))
    return states


def arena_step(envs: dict[str, VacuumCleaningEnv], action: int | str | None = None) -> list[dict[str, Any]]:
    states: list[dict[str, Any]] = []
    for algorithm, env in envs.items():
        states.append(step_env(env, algorithm=algorithm, action=action))
    return states


def normalize_algorithm(algorithm: str) -> str:
    value = algorithm.lower().strip()
    if value not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm '{algorithm}'. Choose from: {', '.join(ALGORITHMS)}")
    return value


def make_env(seed: int | None = None) -> VacuumCleaningEnv:
    env = VacuumCleaningEnv(config_name="config_2.yaml")
    env.reset(seed=seed)
    return env


def normalize_observation(env: VacuumCleaningEnv, observation: np.ndarray) -> np.ndarray:
    low = env.observation_space.low.astype(np.float32)
    high = env.observation_space.high.astype(np.float32)
    scale = high - low
    scale[scale == 0.0] = 1.0
    normalized = 2.0 * ((observation.astype(np.float32) - low) / scale) - 1.0
    return np.clip(normalized, -1.0, 1.0).astype(np.float32)


def continuous_to_discrete_action(env: VacuumCleaningEnv, action: Any) -> int:
    scalar = float(np.asarray(action, dtype=np.float32).reshape(-1)[0])
    scaled = (np.clip(scalar, -1.0, 1.0) + 1.0) * 0.5 * (env.action_space.n - 1)
    return int(np.rint(scaled))


def resolve_action(env: VacuumCleaningEnv, algorithm: str, action: int | str | None) -> tuple[int, str]:
    algorithm = normalize_algorithm(algorithm)
    if action is None or (isinstance(action, str) and action.lower().strip() == "auto"):
        model = policy_cache.get(algorithm)
        observation = env._state_to_obs(env._current_state)
        policy_observation = normalize_observation(env, observation)
        predicted_action, _ = model.predict(policy_observation, deterministic=True)
        if ALGORITHMS[algorithm].continuous_policy:
            action_index = continuous_to_discrete_action(env, predicted_action)
        else:
            action_index = int(predicted_action)
        return action_index, env.ACTIONS[action_index]

    if isinstance(action, str):
        normalized = action.strip().lower()
        action_lookup = {name.lower(): index for index, name in enumerate(env.ACTIONS)}
        if normalized not in action_lookup:
            raise ValueError(f"Unknown action '{action}'. Choose from: {', '.join(env.ACTIONS)}")
        action_index = action_lookup[normalized]
    else:
        action_index = int(action)

    if action_index < 0 or action_index >= env.action_space.n:
        raise ValueError(f"Invalid action index {action_index}. Must be in [0, {env.action_space.n - 1}]")
    return action_index, env.ACTIONS[action_index]


def reset_env(env: VacuumCleaningEnv, algorithm: str = "ppo", seed: int | None = None) -> dict[str, Any]:
    observation, info = env.reset(seed=seed)
    return serialize_env_state(
        env=env,
        algorithm=normalize_algorithm(algorithm),
        observation=observation,
        reward=0.0,
        action_index=None,
        action_name=None,
        terminated=False,
        truncated=False,
        info=info,
    )


def step_env(env: VacuumCleaningEnv, algorithm: str, action: int | str | None = None) -> dict[str, Any]:
    algorithm = normalize_algorithm(algorithm)
    action_index, action_name = resolve_action(env, algorithm, action)
    observation, reward, terminated, truncated, info = env.step(action_index)
    if terminated or truncated:
        info = {**info, "episode_finished": True}
    return serialize_env_state(
        env=env,
        algorithm=algorithm,
        observation=observation,
        reward=reward,
        action_index=action_index,
        action_name=action_name,
        terminated=terminated,
        truncated=truncated,
        info=info,
    )


def serialize_env_state(
    *,
    env: VacuumCleaningEnv,
    algorithm: str,
    observation: np.ndarray,
    reward: float,
    action_index: int | None,
    action_name: str | None,
    terminated: bool,
    truncated: bool,
    info: dict[str, Any],
) -> dict[str, Any]:
    state = env._current_state
    if state is None:
        raise RuntimeError("Environment has not been reset.")

    grid: list[list[int]] = []
    for row in range(1, env.grid_rows + 1):
        grid_row: list[int] = []
        for col in range(1, env.grid_cols + 1):
            grid_row.append(int(state.dirt[env.cell_to_idx[(row, col)]]))
        grid.append(grid_row)

    adjacent = env._adjacent_status_vector(state)
    serializable_info = {
        key: _to_jsonable(value)
        for key, value in info.items()
        if key != "state"
    }
    battery_reset = bool(serializable_info.get("manual_reset") or serializable_info.get("battery_depleted"))
    relocated_to_dock = battery_reset and state.robot_pos == env.dock_pos
    return {
        "algorithm": algorithm,
        "observation": [float(value) for value in np.asarray(observation).tolist()],
        "robot_pos": [int(state.robot_pos[0]), int(state.robot_pos[1])],
        "battery": int(state.battery),
        "battery_capacity": int(env.battery_capacity),
        "grid": grid,
        "obstacles": [[int(row), int(col)] for row, col in state.obstacles],
        "dock": [int(env.dock_pos[0]), int(env.dock_pos[1])],
        "adjacent_cells": [int(value) for value in adjacent],
        "action": action_name,
        "action_index": action_index,
        "reward": float(reward),
        "step_count": int(env._step_count),
        "episode_return": float(env._episode_return),
        "total_dirt": int(sum(state.dirt)),
        "battery_reset": battery_reset,
        "relocated_to_dock": relocated_to_dock,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "info": serializable_info,
    }


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    return value
