from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import yaml
from gymnasium import error, spaces
from gymnasium.envs.registration import register, registry

Coord = Tuple[int, int]
ConfigDict = Dict[str, Any]

CONFIG_DIR = Path(__file__).resolve().parent / "config"
ENV_CONFIG_MAP = {
    "VacuumCleaner-v1": "config_1.yaml",
    "VacuumCleaner-v2": "config_2.yaml",
}


@dataclass(frozen=True)
class State:
    robot_pos: Coord
    battery: int
    dirt: Tuple[int, ...]
    obstacles: Tuple[Coord, ...]
    is_failure: bool = False


def _load_yaml_file(path: Path) -> ConfigDict:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")
    return data


def _deep_merge(base: ConfigDict, override: Optional[ConfigDict]) -> ConfigDict:
    merged = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_env_config(config_name: Optional[str] = None, config_path: Optional[str | Path] = None) -> ConfigDict:
    if config_path is not None:
        path = Path(config_path)
    else:
        selected_name = config_name or "config_1.yaml"
        path = Path(selected_name)
        if not path.is_absolute():
            path = CONFIG_DIR / path
    if not path.exists():
        raise FileNotFoundError(f"Environment config not found: {path}")
    return _load_yaml_file(path)


def register_env_variant(env_id: str, config_name: str) -> None:
    if env_id in registry:
        return
    register(
        id=env_id,
        entry_point="env:VacuumCleaningEnv",
        kwargs={"config_name": config_name},
        order_enforce=False,
        disable_env_checker=True,
    )


def ensure_env_registered() -> None:
    for env_id, config_name in ENV_CONFIG_MAP.items():
        register_env_variant(env_id, config_name)


class VacuumCleaningEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array", "ansi"], "render_fps": 8}

    ACTIONS = ["N", "S", "E", "W", "Stay", "Clean", "Charge"]
    MOVE_ACTIONS = {"N", "S", "E", "W"}
    STATUS_BLOCKED = 0
    STATUS_DOCK = 1
    STATUS_CLEAN = 2

    def __init__(
        self,
        config: Optional[ConfigDict] = None,
        config_name: Optional[str] = None,
        config_path: Optional[str | Path] = None,
        render_mode: Optional[str] = None,
    ):
        if render_mode is not None and render_mode not in self.metadata.get("render_modes", []):
            raise ValueError(
                f"Invalid render_mode '{render_mode}'. "
                f"Supported modes: {self.metadata.get('render_modes', [])}"
            )
        self.render_mode = render_mode

        yaml_config = load_env_config(config_name=config_name, config_path=config_path)
        self.config = _deep_merge(yaml_config, config)

        self.grid_rows = int(self.config["grid_rows"])
        self.grid_cols = int(self.config["grid_cols"])
        self.battery_capacity = int(self.config.get("battery_capacity", 100))
        self.charge_amount = int(self.config.get("charge_amount", 5))
        self._max_steps = int(self.config.get("max_steps", 500))
        self.random_seed = int(self.config.get("random_seed", 12345))

        self.cells: List[Coord] = [
            (x, y) for x in range(1, self.grid_rows + 1) for y in range(1, self.grid_cols + 1)
        ]
        self.cell_to_idx: Dict[Coord, int] = {cell: idx for idx, cell in enumerate(self.cells)}

        dock_cfg = self.config.get("dock_position", [1, 1])
        self.dock_pos = tuple(dock_cfg)
        if self.dock_pos not in self.cells:
            raise ValueError(f"dock_position {self.dock_pos} is outside the grid.")

        self.possible_obstacle_sets = self._build_possible_obstacle_sets(self.config.get("obstacle", {}))
        self.obstacles = self.possible_obstacle_sets[0]
        self._set_episode_obstacles(self.obstacles)

        self.p_move = float(self.config.get("p_move", 0.9))
        dirt_cfg = self.config.get("dirt_dynamics", {})
        self.rho = np.array(
            [
                float(dirt_cfg.get("rho0", self.config.get("rho0", 0.20))),
                float(dirt_cfg.get("rho1", self.config.get("rho1", 0.10))),
                float(dirt_cfg.get("rho2", self.config.get("rho2", 0.05))),
                0.0,
            ],
            dtype=float,
        )
        self.rho = np.clip(self.rho, 0.0, 1.0)
        # self.rho = np.zeros_like(self.rho)

        reward_cfg = self.config.get("rewards", {})
        self.step_cost = float(reward_cfg.get("step_cost", 1.0))
        self.move_cost = float(reward_cfg.get("move_cost", 1.0))
        self.slip_penalty = float(reward_cfg.get("slip_penalty", 2.0))
        self.bump_penalty = float(reward_cfg.get("bump_penalty", 5.0))
        self.idle_penalty = float(reward_cfg.get("idle_penalty", 2.0))
        self.clean_cost = float(reward_cfg.get("clean_cost", 2.0))
        self.waste_penalty = float(reward_cfg.get("waste_penalty", 4.0))
        self.dock_clean_penalty = float(reward_cfg.get("dock_clean_penalty", 8.0))
        self.charge_cost = float(reward_cfg.get("charge_cost", 1.0))
        self.full_charge_penalty = float(reward_cfg.get("full_charge_penalty", 5.0))
        self.away_charge_penalty = float(reward_cfg.get("away_charge_penalty", 10.0))
        self.clean_rewards = {
            1: float(reward_cfg.get("alpha_1", 8.0)),
            2: float(reward_cfg.get("alpha_2", 12.0)),
            3: float(reward_cfg.get("alpha_3", 16.0)),
        }
        self.charge_reward_scale = float(reward_cfg.get("alpha_charge", 0.8))
        self.return_weight = float(reward_cfg.get("lambda_return", 2.0))
        self.low_battery_threshold = int(reward_cfg.get("low_battery_threshold", 25))
        self.reset_penalty = float(reward_cfg.get("reset_penalty", 300.0))

        q_cfg = self.config.get("initial_dirt_probs")
        if q_cfg:
            q = np.array([q_cfg.get(f"q{i}", 0.0) for i in range(4)], dtype=float)
            q_sum = float(np.sum(q))
            self.initial_dirt_probs = (q / q_sum) if q_sum > 0 else np.full(4, 0.25)
        else:
            self.initial_dirt_probs = np.full(4, 0.25)

        self.observation_space = self._build_observation_space()
        self.action_space = spaces.Discrete(len(self.ACTIONS))

        self._current_state: Optional[State] = None
        self._step_count = 0
        self._transition_cache: Dict[Tuple[State, str], List[Tuple[float, State, float]]] = {}
        self._window = None
        self._clock = None
        self._font = None
        self._small_font = None
        self._title_font = None
        self._mono_font = None
        self._pygame = None
        self._previous_robot_pos: Optional[Coord] = None
        self._previous_battery: Optional[int] = None
        self._previous_step_count: Optional[int] = None
        self._last_action_name: Optional[str] = None
        self._last_reward = 0.0
        self._episode_return = 0.0
        self._last_manual_reset = False
        self._last_boom_pos: Optional[Coord] = None
        self._last_move_case = "none"
        self._last_collision_pos: Optional[Coord] = None

    def _build_possible_obstacle_sets(self, obstacle_cfg: ConfigDict) -> List[Tuple[Coord, ...]]:
        mode = obstacle_cfg.get("mode", "sampled")
        count = int(obstacle_cfg.get("count", 1))
        if mode == "fixed":
            if "positions" in obstacle_cfg:
                layouts = [tuple(tuple(pos) for pos in obstacle_cfg["positions"])]
            else:
                layouts = [(tuple(obstacle_cfg.get("position", [self.grid_rows, self.grid_cols])),)]
        elif mode == "sampled":
            if "layouts" in obstacle_cfg:
                layouts = [tuple(tuple(pos) for pos in layout) for layout in obstacle_cfg["layouts"]]
            elif "positions" in obstacle_cfg and count == 1:
                layouts = [(tuple(pos),) for pos in obstacle_cfg["positions"]]
            else:
                candidates = [cell for cell in self.cells if cell != self.dock_pos]
                layouts = []
                if count == 1:
                    layouts = [(cell,) for cell in candidates]
                else:
                    import itertools

                    layouts = [tuple(combo) for combo in itertools.combinations(candidates, count)]
        else:
            raise ValueError(f"Unsupported obstacle mode: {mode}")

        valid_layouts: List[Tuple[Coord, ...]] = []
        for layout in layouts:
            unique = tuple(sorted(set(tuple(pos) for pos in layout)))
            if len(unique) != len(layout):
                raise ValueError("Obstacle positions in one layout must be unique.")
            if len(unique) != count and mode == "sampled":
                raise ValueError(f"Expected {count} obstacles, got {len(unique)}.")
            for position in unique:
                if position == self.dock_pos:
                    raise ValueError("Obstacle cannot overlap with dock.")
                if position not in self.cells:
                    raise ValueError(f"Obstacle position {position} is outside the grid.")
            valid_layouts.append(unique)
        if not valid_layouts:
            raise ValueError("At least one obstacle layout must be provided.")
        return valid_layouts

    def _build_observation_space(self) -> spaces.Box:
        local_view_size = 9
        low = [0, 0, 0] + [0] * local_view_size + [-3, -3]
        high = [self.grid_rows, self.grid_cols, self.battery_capacity] + [5] * local_view_size + [3, 3]
        return spaces.Box(
            low=np.array(low, dtype=np.float32),
            high=np.array(high, dtype=np.float32),
            dtype=np.float32,
        )

    def _set_episode_obstacles(self, obstacles: Tuple[Coord, ...]) -> None:
        self.obstacles = tuple(sorted(obstacles))
        self.obstacle_set = set(self.obstacles)
        self.traversable_cells = [cell for cell in self.cells if cell not in self.obstacle_set]
        self.variable_cells = [cell for cell in self.cells if cell != self.dock_pos and cell not in self.obstacle_set]

    def _make_state(
        self,
        robot_pos: Coord,
        battery: int,
        dirt: Tuple[int, ...],
        obstacles: Optional[Tuple[Coord, ...]] = None,
        is_failure: bool = False,
    ) -> State:
        if is_failure:
            return self._failure_state()
        obstacles = self.obstacles if obstacles is None else tuple(sorted(obstacles))
        obstacle_set = set(obstacles)
        if robot_pos not in self.cells:
            raise ValueError(f"Robot position {robot_pos} is outside the grid.")
        if robot_pos in obstacle_set:
            raise ValueError("Robot cannot overlap with obstacle.")
        if len(dirt) != len(self.cells):
            raise ValueError("Dirt tuple length must match number of grid cells.")

        dirt_list = list(dirt)
        dirt_list[self.cell_to_idx[self.dock_pos]] = 0
        for obstacle in obstacles:
            dirt_list[self.cell_to_idx[obstacle]] = 0

        return State(
            robot_pos=robot_pos,
            battery=int(np.clip(battery, 0, self.battery_capacity)),
            dirt=tuple(int(level) for level in dirt_list),
            obstacles=obstacles,
            is_failure=False,
        )

    @staticmethod
    def _failure_state() -> State:
        return State((0, 0), 0, tuple(), tuple(), True)

    def _is_inside(self, coord: Coord) -> bool:
        x, y = coord
        return 1 <= x <= self.grid_rows and 1 <= y <= self.grid_cols

    def enumerate_all_states(self) -> List[State]:
        all_states: List[State] = []
        for obstacles in self.possible_obstacle_sets:
            obstacle_set = set(obstacles)
            traversable = [cell for cell in self.cells if cell not in obstacle_set]
            variable = [cell for cell in self.cells if cell != self.dock_pos and cell not in obstacle_set]
            for robot_pos in traversable:
                for battery in range(self.battery_capacity + 1):
                    for dirt_levels in np.ndindex(*(4 for _ in variable)):
                        dirt = [0] * len(self.cells)
                        for cell, level in zip(variable, dirt_levels):
                            dirt[self.cell_to_idx[cell]] = int(level)
                        all_states.append(self._make_state(robot_pos, battery, tuple(dirt), obstacles))
        return all_states

    def _state_to_obs(self, state: State) -> np.ndarray:
        if state.is_failure:
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        x, y = state.robot_pos
        z = self._adjacent_status_vector(state)
        rd_x, rd_y = self._scaled_relative_dock_distance(state.robot_pos)
        return np.array([x, y, state.battery, *z, rd_x, rd_y], dtype=np.float32)

    def _adjacent_status_vector(self, state: State) -> List[int]:
        x, y = state.robot_pos
        positions = [
            (x, y),
            (x - 1, y),
            (x, y + 1),
            (x + 1, y),
            (x, y - 1),
            (x - 1, y - 1),
            (x - 1, y + 1),
            (x + 1, y + 1),
            (x + 1, y - 1),
        ]
        return [self._cell_status(pos, state) for pos in positions]

    def _cell_status(self, pos: Coord, state: State) -> int:
        if not self._is_inside(pos) or pos in set(state.obstacles):
            return self.STATUS_BLOCKED
        if pos == self.dock_pos:
            return self.STATUS_DOCK
        return self.STATUS_CLEAN + int(state.dirt[self.cell_to_idx[pos]])

    def _scaled_relative_dock_distance(self, pos: Coord) -> Tuple[int, int]:
        dx = self.dock_pos[0] - pos[0]
        dy = self.dock_pos[1] - pos[1]
        return (
            self._scale_relative_distance(dx, max(self.grid_rows - 1, 1)),
            self._scale_relative_distance(dy, max(self.grid_cols - 1, 1)),
        )

    @staticmethod
    def _scale_relative_distance(delta: int, denominator: int) -> int:
        if delta == 0:
            return 0
        return int(np.sign(delta) * np.ceil(3.0 * abs(delta) / denominator))

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        if seed is not None:
            super().reset(seed=seed)
        elif getattr(self, "np_random", None) is None:
            super().reset(seed=self.random_seed)

        obstacles = self.possible_obstacle_sets[int(self.np_random.integers(0, len(self.possible_obstacle_sets)))]
        self._set_episode_obstacles(obstacles)

        battery = int(self.np_random.integers(0, self.battery_capacity + 1))
        dirt = [0] * len(self.cells)
        for cell in self.variable_cells:
            dirt[self.cell_to_idx[cell]] = int(self.np_random.choice([0, 1, 2, 3], p=self.initial_dirt_probs))

        self._current_state = self._make_state(self.dock_pos, battery, tuple(dirt), self.obstacles)
        self._step_count = 0
        self._previous_robot_pos = None
        self._previous_battery = None
        self._previous_step_count = None
        self._last_action_name = None
        self._last_reward = 0.0
        self._episode_return = 0.0
        self._last_manual_reset = False
        self._last_boom_pos = None
        self._last_move_case = "none"
        self._last_collision_pos = None
        if self.render_mode == "human":
            self.render()
        return self._state_to_obs(self._current_state), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        if self._current_state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        if not isinstance(action, (int, np.integer)):
            raise TypeError(f"Action must be int, got {type(action)}")
        if action < 0 or action >= len(self.ACTIONS):
            raise ValueError(f"Invalid action {action}. Must be in [0, {len(self.ACTIONS) - 1}]")

        action_name = self.ACTIONS[int(action)]
        previous_state = self._current_state
        self._previous_battery = previous_state.battery
        self._previous_step_count = self._step_count
        next_state, reward, manual_reset, boom_pos, move_case, collision_pos = self._step_mpi(self._current_state, action_name)
        self._current_state = next_state
        if not next_state.is_failure:
            self._set_episode_obstacles(next_state.obstacles)
        self._step_count += 1
        self._previous_robot_pos = None if previous_state.is_failure else previous_state.robot_pos
        self._last_action_name = action_name
        self._last_reward = float(reward)
        self._episode_return += float(reward)
        self._last_manual_reset = bool(manual_reset)
        self._last_boom_pos = boom_pos
        self._last_move_case = move_case
        self._last_collision_pos = collision_pos

        terminated = next_state.is_failure
        truncated = self._step_count >= self._max_steps
        info = {
            "requested_action": action_name,
            "action": action_name,
            "state": next_state,
            "battery": next_state.battery,
            "total_dirt": sum(next_state.dirt) if not next_state.is_failure else 0,
            "at_dock": next_state.robot_pos == self.dock_pos if not next_state.is_failure else False,
            "obstacles": list(next_state.obstacles) if not next_state.is_failure else [],
            "battery_depleted": bool(manual_reset),
            "manual_reset": bool(manual_reset),
            "move_case": move_case,
            "collision_pos": collision_pos,
        }
        if self.render_mode == "human":
            self.render()
        return self._state_to_obs(next_state), reward, terminated, truncated, info

    def _step_mpi(self, state: State, action: str) -> Tuple[State, float, bool, Optional[Coord], str, Optional[Coord]]:
        if state.is_failure:
            return self._failure_state(), 0.0, False, None, "none", None

        battery_after = self._next_battery(state, action)
        clean_dirt = self._apply_cleaning(state.dirt, state.robot_pos, action, state.obstacles)
        dirt_next = self._sample_dirt_spontaneous_evolution(clean_dirt, state.obstacles)

        movement_outcomes = self._movement_outcomes(state.robot_pos, action, state.obstacles)
        move_probs = np.array([prob for prob, _, _ in movement_outcomes], dtype=float)
        move_probs /= move_probs.sum()
        move_idx = int(self.np_random.choice(len(movement_outcomes), p=move_probs))
        _, next_pos, move_case = movement_outcomes[move_idx]
        collision_pos = self._target_for_action(state.robot_pos, action) if move_case == "bump" else None

        manual_reset = (battery_after == 0) and (next_pos != self.dock_pos)
        boom_pos = next_pos if manual_reset else None
        final_pos = self.dock_pos if manual_reset else next_pos
        next_state = self._make_state(final_pos, battery_after, dirt_next, state.obstacles)
        reward = self._reward(
            state=state,
            action=action,
            next_state=next_state,
            move_case=move_case,
            old_distance=self._dock_distance_score(state.robot_pos),
            manual_reset=manual_reset,
        )
        return next_state, float(reward), bool(manual_reset), boom_pos, move_case, collision_pos

    def _get_transition_distribution(self, state: State, action: str) -> List[Tuple[float, State, float]]:
        if state.is_failure:
            return [(1.0, self._failure_state(), 0.0)]

        cache_key = (state, action)
        if cache_key in self._transition_cache:
            return self._transition_cache[cache_key]

        battery_after = self._next_battery(state, action)
        clean_dirt = self._apply_cleaning(state.dirt, state.robot_pos, action, state.obstacles)
        dirt_dist = self._dirt_spontaneous_evolution(clean_dirt, state.obstacles)

        outcomes: List[Tuple[float, State, float]] = []
        old_distance = self._dock_distance_score(state.robot_pos)
        for move_prob, next_pos, move_case in self._movement_outcomes(state.robot_pos, action, state.obstacles):
            for dirt_next, dirt_prob in dirt_dist.items():
                manual_reset = (battery_after == 0) and (next_pos != self.dock_pos)
                final_pos = self.dock_pos if manual_reset else next_pos
                next_state = self._make_state(final_pos, battery_after, dirt_next, state.obstacles)
                reward = self._reward(
                    state=state,
                    action=action,
                    next_state=next_state,
                    move_case=move_case,
                    old_distance=old_distance,
                    manual_reset=manual_reset,
                )
                outcomes.append((move_prob * dirt_prob, next_state, reward))

        total_prob = sum(prob for prob, _, _ in outcomes)
        result = [(prob / total_prob, next_state, reward) for prob, next_state, reward in outcomes]
        self._transition_cache[cache_key] = result
        return result

    def _next_battery(self, state: State, action: str) -> int:
        if action == "Charge":
            if state.robot_pos == self.dock_pos:
                return min(state.battery + self.charge_amount, self.battery_capacity)
            return state.battery
        if action == "Clean":
            return max(state.battery - 2, 0)
        return max(state.battery - 1, 0)

    def _movement_outcomes(
        self,
        pos: Coord,
        action: str,
        obstacles: Tuple[Coord, ...],
    ) -> List[Tuple[float, Coord, str]]:
        if action not in self.MOVE_ACTIONS:
            return [(1.0, pos, "none")]

        x, y = pos
        if action == "N":
            target = (x - 1, y)
        elif action == "S":
            target = (x + 1, y)
        elif action == "E":
            target = (x, y + 1)
        else:
            target = (x, y - 1)

        if not self._is_inside(target) or target in set(obstacles):
            return [(1.0, pos, "bump")]
        return [(self.p_move, target, "success"), (1.0 - self.p_move, pos, "slip")]

    def _target_for_action(self, pos: Coord, action: str) -> Coord:
        x, y = pos
        if action == "N":
            return (x - 1, y)
        if action == "S":
            return (x + 1, y)
        if action == "E":
            return (x, y + 1)
        if action == "W":
            return (x, y - 1)
        return pos

    def _apply_cleaning(
        self,
        dirt: Tuple[int, ...],
        robot_pos: Coord,
        action: str,
        obstacles: Tuple[Coord, ...],
    ) -> Tuple[int, ...]:
        if action != "Clean" or robot_pos == self.dock_pos or robot_pos in set(obstacles):
            return dirt
        idx = self.cell_to_idx[robot_pos]
        out = list(dirt)
        out[idx] = max(int(out[idx]) - 1, 0)
        return tuple(out)

    def _dirt_spontaneous_evolution(self, dirt: Tuple[int, ...], obstacles: Tuple[Coord, ...]) -> Dict[Tuple[int, ...], float]:
        dist: Dict[Tuple[int, ...], float] = {dirt: 1.0}
        variable_cells = [cell for cell in self.cells if cell != self.dock_pos and cell not in set(obstacles)]
        for cell in variable_cells:
            idx = self.cell_to_idx[cell]
            next_dist: Dict[Tuple[int, ...], float] = {}
            for current_dirt, prob in dist.items():
                level = int(current_dirt[idx])
                for next_level, trans_prob in self._tile_transition_probs(level).items():
                    mod = list(current_dirt)
                    mod[idx] = next_level
                    mod_t = tuple(mod)
                    next_dist[mod_t] = next_dist.get(mod_t, 0.0) + prob * trans_prob
            dist = next_dist

        final: Dict[Tuple[int, ...], float] = {}
        for current_dirt, prob in dist.items():
            mod = list(current_dirt)
            mod[self.cell_to_idx[self.dock_pos]] = 0
            for obstacle in obstacles:
                mod[self.cell_to_idx[obstacle]] = 0
            mod_t = tuple(mod)
            final[mod_t] = final.get(mod_t, 0.0) + prob
        return final

    def _sample_dirt_spontaneous_evolution(
        self,
        dirt: Tuple[int, ...],
        obstacles: Tuple[Coord, ...],
    ) -> Tuple[int, ...]:
        obstacle_set = set(obstacles)
        out = list(dirt)
        for cell in self.cells:
            if cell == self.dock_pos or cell in obstacle_set:
                out[self.cell_to_idx[cell]] = 0
                continue
            idx = self.cell_to_idx[cell]
            level = int(out[idx])
            if level in {0, 1, 2} and self.np_random.random() < float(self.rho[level]):
                out[idx] = level + 1
        return tuple(out)

    def _tile_transition_probs(self, level: int) -> Dict[int, float]:
        if level in {0, 1, 2}:
            return {level: 1.0 - float(self.rho[level]), level + 1: float(self.rho[level])}
        return {3: 1.0}

    def _reward(
        self,
        state: State,
        action: str,
        next_state: State,
        move_case: str,
        old_distance: int,
        manual_reset: bool = False,
    ) -> float:
        reward = -self.step_cost

        if action in self.MOVE_ACTIONS:
            reward -= self.move_cost
            if move_case == "slip":
                reward -= self.slip_penalty
            elif move_case == "bump":
                reward -= self.bump_penalty
        elif action == "Stay":
            reward -= self.idle_penalty
        elif action == "Clean":
            current_status = self._cell_status(state.robot_pos, state)
            if current_status == self.STATUS_DOCK:
                reward -= self.clean_cost + self.dock_clean_penalty
            else:
                dirt_level = int(state.dirt[self.cell_to_idx[state.robot_pos]])
                if dirt_level > 0:
                    reward += self.clean_rewards[dirt_level] - self.clean_cost
                else:
                    reward -= self.clean_cost + self.waste_penalty
        elif action == "Charge":
            if state.robot_pos == self.dock_pos:
                if state.battery >= self.battery_capacity:
                    reward -= self.charge_cost + self.full_charge_penalty
                else:
                    recovered = min(state.battery + self.charge_amount, self.battery_capacity) - state.battery
                    reward += self.charge_reward_scale * recovered - self.charge_cost
            else:
                reward -= self.charge_cost + self.away_charge_penalty

        if manual_reset or next_state.is_failure:
            reward -= self.reset_penalty
        elif state.battery <= self.low_battery_threshold:
            reward += self.return_weight * (old_distance - self._dock_distance_score(next_state.robot_pos))

        return float(reward)

    def _dock_distance_score(self, pos: Coord) -> int:
        rd_x, rd_y = self._scaled_relative_dock_distance(pos)
        return abs(rd_x) + abs(rd_y)

    def _load_pygame(self):
        if self._pygame is not None:
            return self._pygame
        try:
            import pygame
        except ImportError as exc:
            raise error.DependencyNotInstalled(
                "pygame is required for render_mode='human' and render_mode='rgb_array'. "
                "Install it with `pip install pygame`."
            ) from exc
        self._pygame = pygame
        return pygame

    def _render_ansi(self) -> str:
        if self._current_state is None:
            return "Environment not initialized. Call reset() first.\n"

        state = self._current_state
        rows = []
        for row in range(1, self.grid_rows + 1):
            tokens = []
            for col in range(1, self.grid_cols + 1):
                pos = (row, col)
                if pos == state.robot_pos:
                    token = "R"
                elif pos == self.dock_pos:
                    token = "D"
                elif pos in set(state.obstacles):
                    token = "O"
                else:
                    level = int(state.dirt[self.cell_to_idx[pos]])
                    token = "." if level == 0 else str(level)
                tokens.append(token)
            rows.append(" ".join(tokens))
        return (
            "\n".join(rows)
            + f"\nstep={self._step_count} battery={state.battery} "
            + f"total_dirt={sum(state.dirt)}\n"
        )

    def _render_frame(
        self,
        robot_progress: float = 1.0,
        ui_progress: float = 1.0,
        boom_phase: float = 1.0,
        collision_phase: float = 1.0,
        show_home: bool = False,
    ) -> np.ndarray:
        if self._current_state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        pygame = self._load_pygame()
        pygame.init()
        if self._font is None:
            pygame.font.init()
            self._font = pygame.font.SysFont("segoeui", 18, bold=True)
            self._small_font = pygame.font.SysFont("segoeui", 13)
            self._title_font = pygame.font.SysFont("segoeui", 24, bold=True)
            self._mono_font = pygame.font.SysFont("consolas", 14, bold=True)

        cell_size = 76
        margin = 24
        header_height = 72
        side_panel_width = 328
        grid_width = self.grid_cols * cell_size
        grid_height = self.grid_rows * cell_size
        width = margin * 3 + grid_width + side_panel_width
        height = margin * 2 + header_height + grid_height + 280
        surface = pygame.Surface((width, height), pygame.SRCALPHA)
        surface.fill((235, 239, 245))

        state = self._current_state
        robot_progress = float(np.clip(robot_progress, 0.0, 1.0))
        ui_progress = float(np.clip(ui_progress, 0.0, 1.0))
        boom_phase = float(np.clip(boom_phase, 0.0, 1.0))
        collision_phase = float(np.clip(collision_phase, 0.0, 1.0))
        colors = {
            "bg": (235, 239, 245),
            "board": (246, 248, 251),
            "tile": (255, 255, 255),
            "tile_alt": (249, 250, 252),
            "tile_seen": (225, 239, 255),
            "grid": (198, 208, 222),
            "text": (17, 24, 39),
            "muted": (96, 108, 128),
            "subtle": (148, 163, 184),
            "dock": (216, 248, 226),
            "dock_line": (48, 163, 93),
            "robot": (32, 102, 209),
            "robot_dark": (20, 64, 145),
            "dry": (214, 121, 35),
            "dry_bg": (255, 247, 237),
            "wet": (20, 143, 203),
            "wet_bg": (240, 249, 255),
            "sticky": (209, 69, 76),
            "sticky_bg": (255, 241, 242),
            "clean": (255, 255, 255),
            "obstacle": (30, 41, 59),
            "obstacle_dot": (71, 85, 105),
            "panel": (255, 255, 255),
            "panel_soft": (248, 250, 252),
            "panel_line": (220, 227, 236),
            "battery": (30, 170, 91),
            "warning": (217, 119, 6),
            "danger": (220, 38, 38),
            "accent": (67, 83, 255),
            "accent_soft": (232, 235, 255),
        }

        def draw_shadow(rect, radius=14, offset=(0, 4), alpha=30):
            shadow = pygame.Surface((rect.width + 10, rect.height + 10), pygame.SRCALPHA)
            shadow_rect = pygame.Rect(5 + offset[0], 5 + offset[1], rect.width, rect.height)
            pygame.draw.rect(shadow, (15, 23, 42, alpha), shadow_rect, border_radius=radius)
            surface.blit(shadow, (rect.left - 5, rect.top - 5))

        def draw_text(text, pos, color=None, font=None):
            font = font or self._small_font
            color = color or colors["text"]
            label = font.render(text, True, color)
            surface.blit(label, pos)

        def draw_centered_text(text, rect, color=None, font=None):
            font = font or self._small_font
            color = color or colors["text"]
            label = font.render(text, True, color)
            surface.blit(label, label.get_rect(center=rect.center))

        def draw_card(rect, title, value, value_color=None, value_y_offset=31):
            pygame.draw.rect(surface, colors["panel_soft"], rect, border_radius=9)
            pygame.draw.rect(surface, colors["panel_line"], rect, width=1, border_radius=9)
            draw_text(title, (rect.left + 12, rect.top + 8), colors["muted"])
            draw_text(value, (rect.left + 12, rect.top + value_y_offset), value_color or colors["text"], self._font)

        def draw_section_label(text, y_pos):
            draw_text(text, (panel.left + 18, y_pos), colors["text"], self._font)
            pygame.draw.line(
                surface,
                colors["panel_line"],
                (panel.left + 18, y_pos + 27),
                (panel.right - 18, y_pos + 27),
                1,
            )

        def draw_pill(rect, text, fill, color):
            pygame.draw.rect(surface, fill, rect, border_radius=rect.height // 2)
            draw_centered_text(text, rect, color, self._small_font)

        def draw_dock(rect):
            pygame.draw.rect(surface, colors["dock"], rect, border_radius=9)
            pygame.draw.rect(surface, colors["dock_line"], rect, width=2, border_radius=9)
            body_width = max(12, int(rect.width * 0.40))
            body_height = max(16, int(rect.height * 0.50))
            body = pygame.Rect(0, 0, body_width, body_height)
            body.center = (rect.centerx, rect.centery + 2)
            pygame.draw.rect(surface, (255, 255, 255), body, border_radius=4)
            pygame.draw.rect(surface, colors["dock_line"], body, width=2, border_radius=4)
            pygame.draw.polygon(
                surface,
                colors["battery"],
                [
                    (body.centerx + 2, body.top + 5),
                    (body.centerx - 6, body.centery + 2),
                    (body.centerx, body.centery + 2),
                    (body.centerx - 4, body.bottom - 5),
                    (body.centerx + 7, body.centery - 2),
                    (body.centerx + 1, body.centery - 2),
                ],
            )

        def draw_obstacle(rect):
            pygame.draw.rect(surface, colors["obstacle"], rect, border_radius=9)
            for dx, dy, radius in [(-14, -10, 5), (7, -12, 4), (12, 9, 7), (-6, 10, 4)]:
                pygame.draw.circle(surface, colors["obstacle_dot"], (rect.centerx + dx, rect.centery + dy), radius)

        def draw_robot(rect, shadow=True, center_override=None):
            center = center_override or rect.center
            radius = max(14, rect.width // 4)
            if shadow:
                pygame.draw.circle(surface, (145, 158, 178), (center[0] + 2, center[1] + 4), radius + 3)
            pygame.draw.circle(surface, (219, 234, 254), center, radius + 6)
            pygame.draw.circle(surface, colors["robot"], center, radius)
            pygame.draw.circle(surface, (255, 255, 255), center, max(5, radius // 3))
            pygame.draw.circle(surface, colors["robot_dark"], center, radius, width=3)
            pygame.draw.circle(surface, (255, 255, 255), (center[0], center[1] - radius + 5), max(2, radius // 7))
            if self._previous_robot_pos:
                px, py = self._previous_robot_pos
                cx, cy = state.robot_pos
                direction = (cy - py, cx - px)
                tip = (center[0] + direction[0] * 9, center[1] + direction[1] * 9)
                pygame.draw.circle(surface, (255, 255, 255), tip, 3)

        def draw_boom(rect, phase=1.0):
            center = rect.center
            scale = 0.92 + 0.22 * np.sin(np.pi * phase)
            rect = rect.inflate(int(rect.width * (scale - 1.0)), int(rect.height * (scale - 1.0)))
            burst = [
                (center[0], rect.top + 1),
                (center[0] + 9, center[1] - 15),
                (rect.right - 3, center[1] - 12),
                (center[0] + 16, center[1]),
                (rect.right - 1, center[1] + 13),
                (center[0] + 7, center[1] + 12),
                (center[0], rect.bottom - 1),
                (center[0] - 8, center[1] + 12),
                (rect.left + 1, center[1] + 13),
                (center[0] - 16, center[1]),
                (rect.left + 3, center[1] - 12),
                (center[0] - 9, center[1] - 15),
            ]
            alpha_ring = max(30, int(150 * (1.0 - phase)))
            pulse = pygame.Surface((rect.width + 34, rect.height + 34), pygame.SRCALPHA)
            pygame.draw.circle(pulse, (239, 68, 68, alpha_ring), (pulse.get_width() // 2, pulse.get_height() // 2), rect.width // 2 + int(16 * phase))
            surface.blit(pulse, (center[0] - pulse.get_width() // 2, center[1] - pulse.get_height() // 2))
            pygame.draw.circle(surface, (248, 113, 113), center, rect.width // 2)
            pygame.draw.polygon(surface, (239, 68, 68), burst)
            pygame.draw.polygon(surface, (255, 214, 10), [p for index, p in enumerate(burst) if index % 2 == 0])
            pygame.draw.circle(surface, (255, 255, 255), center, max(12, rect.width // 5))
            draw_centered_text("BOOM", pygame.Rect(rect.left - 4, center[1] - 10, rect.width + 8, 22), colors["danger"], self._mono_font)

        def draw_collision(rect, phase=1.0):
            center = rect.center
            ring_radius = 12 + int(18 * phase)
            alpha = max(35, int(190 * (1.0 - phase)))
            ring = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
            pygame.draw.circle(ring, (220, 38, 38, alpha), (rect.width // 2, rect.height // 2), ring_radius, width=4)
            surface.blit(ring, rect.topleft)
            jitter = int(4 * np.sin(phase * np.pi * 6))
            spark_color = (255, 214, 10)
            for angle in range(0, 360, 45):
                rad = np.deg2rad(angle)
                inner = (
                    center[0] + int(np.cos(rad) * (8 + jitter)),
                    center[1] + int(np.sin(rad) * (8 + jitter)),
                )
                outer = (
                    center[0] + int(np.cos(rad) * (18 + 10 * phase)),
                    center[1] + int(np.sin(rad) * (18 + 10 * phase)),
                )
                pygame.draw.line(surface, spark_color, inner, outer, width=3)
            badge = pygame.Rect(0, 0, 48, 20)
            badge.center = (center[0], center[1] - 24)
            pygame.draw.rect(surface, (254, 242, 242), badge, border_radius=10)
            pygame.draw.rect(surface, colors["danger"], badge, width=1, border_radius=10)
            draw_centered_text("BUMP", badge, colors["danger"], self._mono_font)

        def draw_legend_marker(rect, kind):
            pygame.draw.rect(surface, colors["panel_soft"], rect, border_radius=7)
            pygame.draw.rect(surface, colors["panel_line"], rect, width=1, border_radius=7)
            center = rect.center
            if kind == "dock":
                pygame.draw.rect(surface, colors["dock"], rect.inflate(-4, -4), border_radius=5)
                pygame.draw.rect(surface, colors["dock_line"], rect.inflate(-4, -4), width=1, border_radius=5)
            elif kind == "robot":
                pygame.draw.circle(surface, colors["robot"], center, 8)
                pygame.draw.circle(surface, (255, 255, 255), center, 3)
            elif kind == "dry":
                pygame.draw.circle(surface, colors["dry"], (center[0] - 5, center[1] - 2), 4)
                pygame.draw.circle(surface, colors["dry"], (center[0] + 5, center[1] + 3), 3)
            elif kind == "wet":
                pygame.draw.circle(surface, colors["wet"], center, 8)
                pygame.draw.polygon(surface, colors["wet"], [(center[0], rect.top + 5), (center[0] - 7, center[1]), (center[0] + 7, center[1])])
            elif kind == "sticky":
                pygame.draw.circle(surface, colors["sticky"], center, 8)
                pygame.draw.circle(surface, colors["sticky"], (center[0] - 5, center[1] + 4), 5)
            elif kind == "obstacle":
                pygame.draw.rect(surface, colors["obstacle"], rect.inflate(-5, -5), border_radius=5)
            else:
                pygame.draw.rect(surface, colors["clean"], rect.inflate(-4, -4), border_radius=5)

        def draw_dirt(rect, level):
            dirt_palettes = {
                1: (colors["dry"], colors["dry_bg"], [(0.30, 0.38, 6), (0.55, 0.30, 4), (0.67, 0.55, 5), (0.43, 0.66, 3)]),
                2: (colors["wet"], colors["wet_bg"], [(0.45, 0.48, 9), (0.58, 0.58, 7), (0.38, 0.61, 5)]),
                3: (colors["sticky"], colors["sticky_bg"], [(0.45, 0.43, 10), (0.58, 0.53, 9), (0.37, 0.60, 7)]),
            }
            if level == 1:
                pygame.draw.rect(surface, colors["dry_bg"], rect, border_radius=10)
                for px, py, radius in dirt_palettes[1][2]:
                    pygame.draw.circle(surface, colors["dry"], (rect.left + int(rect.width * px), rect.top + int(rect.height * py)), radius)
            elif level == 2:
                pygame.draw.rect(surface, colors["wet_bg"], rect, border_radius=10)
                drop = [
                    (rect.centerx, rect.centery - 15),
                    (rect.centerx - 12, rect.centery + 4),
                    (rect.centerx, rect.centery + 17),
                    (rect.centerx + 12, rect.centery + 4),
                ]
                pygame.draw.polygon(surface, colors["wet"], drop)
                pygame.draw.circle(surface, colors["wet"], (rect.centerx, rect.centery + 5), 12)
                pygame.draw.circle(surface, (255, 255, 255), (rect.centerx - 4, rect.centery + 1), 3)
            elif level == 3:
                pygame.draw.rect(surface, colors["sticky_bg"], rect, border_radius=10)
                blob = [
                    (rect.centerx - 15, rect.centery - 5),
                    (rect.centerx - 6, rect.centery - 17),
                    (rect.centerx + 12, rect.centery - 12),
                    (rect.centerx + 17, rect.centery + 4),
                    (rect.centerx + 5, rect.centery + 16),
                    (rect.centerx - 14, rect.centery + 11),
                ]
                pygame.draw.polygon(surface, colors["sticky"], blob)
                pygame.draw.circle(surface, (255, 255, 255), (rect.centerx + 4, rect.centery - 4), 3)
            else:
                pygame.draw.rect(surface, colors["tile"], rect, border_radius=10)

        def draw_tile(pos, rect):
            pygame.draw.rect(surface, (214, 223, 235), rect.move(1, 2), border_radius=11)
            base_color = colors["tile_alt"] if (pos[0] + pos[1]) % 2 == 0 else colors["tile"]
            pygame.draw.rect(surface, base_color, rect, border_radius=11)
            inner = rect.inflate(-9, -9)
            if pos == self._previous_robot_pos and pos != state.robot_pos:
                pygame.draw.rect(surface, colors["tile_seen"], inner, border_radius=10)
            if pos == self.dock_pos:
                draw_dock(inner)
            elif pos in set(state.obstacles):
                draw_obstacle(inner)
            else:
                level = int(state.dirt[self.cell_to_idx[pos]])
                draw_dirt(inner, level)
            pygame.draw.rect(surface, colors["grid"], rect, width=1, border_radius=11)
            robot_is_interpolating = (
                self._previous_robot_pos is not None
                and self._previous_robot_pos != state.robot_pos
                and not self._last_manual_reset
                and robot_progress < 0.999
            )
            if pos == state.robot_pos:
                pygame.draw.rect(surface, colors["accent"], rect.inflate(-4, -4), width=3, border_radius=12)
                if not robot_is_interpolating:
                    draw_robot(inner)

        grid_left = margin
        grid_top = margin + header_height
        panel_left = grid_left + grid_width + margin

        def cell_rect(pos):
            row, col = pos
            return pygame.Rect(
                grid_left + (col - 1) * cell_size,
                grid_top + (row - 1) * cell_size,
                cell_size,
                cell_size,
            )

        draw_text("VacuumCleaner-v2", (margin, 11), colors["text"], self._title_font)
        draw_text("Policy rollout viewer", (margin, 43), colors["muted"], self._small_font)
        mode_rect = pygame.Rect(margin + 168, 39, 82, 24)
        draw_pill(mode_rect, "STATIC DIRT", colors["accent_soft"], colors["accent"])

        grid_bg = pygame.Rect(grid_left - 2, grid_top - 2, grid_width + 4, grid_height + 4)
        draw_shadow(grid_bg, radius=16, alpha=24)
        pygame.draw.rect(surface, colors["board"], grid_bg, border_radius=16)
        pygame.draw.rect(surface, colors["panel_line"], grid_bg, width=1, border_radius=16)
        for row in range(1, self.grid_rows + 1):
            for col in range(1, self.grid_cols + 1):
                rect = cell_rect((row, col))
                draw_tile((row, col), rect)
                if self._last_boom_pos == (row, col):
                    draw_boom(rect.inflate(-12, -12), boom_phase)

        if self._last_move_case == "bump" and self._last_collision_pos is not None:
            collision_pos = self._last_collision_pos
            if self._is_inside(collision_pos):
                collision_rect = cell_rect(collision_pos).inflate(-10, -10)
            else:
                current_rect = cell_rect(state.robot_pos).inflate(-10, -10)
                collision_rect = current_rect.copy()
                target = collision_pos
                if target[0] < 1:
                    collision_rect.centery = current_rect.top + 4
                elif target[0] > self.grid_rows:
                    collision_rect.centery = current_rect.bottom - 4
                elif target[1] < 1:
                    collision_rect.centerx = current_rect.left + 4
                elif target[1] > self.grid_cols:
                    collision_rect.centerx = current_rect.right - 4
            draw_collision(collision_rect, collision_phase)

        if (
            self._previous_robot_pos is not None
            and self._previous_robot_pos != state.robot_pos
            and not self._last_manual_reset
            and robot_progress < 0.999
        ):
            start_rect = cell_rect(self._previous_robot_pos).inflate(-9, -9)
            end_rect = cell_rect(state.robot_pos).inflate(-9, -9)
            start_center = start_rect.center
            end_center = end_rect.center
            moving_center = (
                int(start_center[0] + (end_center[0] - start_center[0]) * robot_progress),
                int(start_center[1] + (end_center[1] - start_center[1]) * robot_progress),
            )
            pygame.draw.line(surface, colors["accent_soft"], start_center, end_center, width=8)
            pygame.draw.line(surface, colors["accent"], start_center, moving_center, width=3)
            draw_robot(end_rect, center_override=moving_center)

        panel = pygame.Rect(panel_left, grid_top, side_panel_width, height - grid_top - margin)
        draw_shadow(panel, radius=16, alpha=24)
        pygame.draw.rect(surface, colors["panel"], panel, border_radius=14)
        pygame.draw.rect(surface, colors["panel_line"], panel, width=1, border_radius=14)

        total_dirt = int(sum(state.dirt))
        shown_battery = state.battery
        shown_step_count = self._step_count
        if self._previous_battery is not None:
            shown_battery = self._previous_battery + (state.battery - self._previous_battery) * ui_progress
        if self._previous_step_count is not None:
            shown_step_count = self._previous_step_count + (self._step_count - self._previous_step_count) * ui_progress
        battery_ratio = shown_battery / max(self.battery_capacity, 1)
        battery_color = colors["battery"] if battery_ratio > 0.25 else colors["danger"]
        progress_ratio = min(1.0, shown_step_count / max(self._max_steps, 1))
        reward_color = colors["battery"] if self._last_reward >= 0 else colors["danger"]
        action_label = self._last_action_name or "Reset"

        y = panel.top + 18
        draw_section_label("Run Status", y)
        is_bump = self._last_move_case == "bump"
        mode_label = "RESET" if self._last_manual_reset else "BUMP" if is_bump else "ACTIVE"
        mode_fill = (254, 242, 242) if self._last_manual_reset or is_bump else (236, 253, 245)
        mode_color = colors["danger"] if self._last_manual_reset or is_bump else colors["battery"]
        draw_pill(pygame.Rect(panel.right - 100, y - 1, 80, 22), mode_label, mode_fill, mode_color)
        y += 40

        card_width = (side_panel_width - 58) // 2
        action_color = colors["danger"] if is_bump else colors["accent"]
        draw_card(pygame.Rect(panel.left + 18, y, card_width, 56), "Last Action", action_label, action_color)
        draw_card(
            pygame.Rect(panel.left + 40 + card_width, y, card_width, 56),
            "Position",
            f"{state.robot_pos[0]}, {state.robot_pos[1]}",
        )
        y += 66
        draw_card(pygame.Rect(panel.left + 18, y, card_width, 56), "Total Dirt", str(total_dirt), colors["warning"])
        draw_card(
            pygame.Rect(panel.left + 40 + card_width, y, card_width, 56),
            "Step",
            f"{int(round(shown_step_count))}/{self._max_steps}",
            colors["text"],
        )
        y += 76

        draw_section_label("Battery & Progress", y)
        y += 42

        draw_text("Episode progress", (panel.left + 18, y), colors["muted"])
        draw_text(f"{int(progress_ratio * 100):>3}%", (panel.right - 56, y), colors["muted"], self._mono_font)
        progress_rect = pygame.Rect(panel.left + 18, y + 22, side_panel_width - 54, 14)
        pygame.draw.rect(surface, (226, 232, 240), progress_rect, border_radius=7)
        progress_fill = progress_rect.copy()
        progress_fill.width = max(2, int(progress_rect.width * progress_ratio))
        pygame.draw.rect(surface, colors["accent"], progress_fill, border_radius=7)
        y += 48

        draw_text("Battery", (panel.left + 18, y), colors["muted"])
        draw_text(f"{int(round(shown_battery))}/{self.battery_capacity}", (panel.right - 76, y), battery_color, self._mono_font)
        battery_rect = pygame.Rect(panel.left + 18, y + 22, side_panel_width - 54, 16)
        pygame.draw.rect(surface, (226, 232, 240), battery_rect, border_radius=8)
        fill_rect = battery_rect.copy()
        fill_rect.width = int(battery_rect.width * battery_ratio)
        pygame.draw.rect(surface, battery_color, fill_rect, border_radius=8)
        if battery_ratio <= 0.25:
            draw_text("Low battery - return to dock", (panel.left + 18, y + 44), colors["danger"])
            y += 64
        else:
            y += 54

        draw_section_label("Rewards", y)
        y += 42

        draw_card(
            pygame.Rect(panel.left + 18, y, side_panel_width - 54, 50),
            "Last Reward",
            f"{self._last_reward:.1f}",
            reward_color,
            value_y_offset=28,
        )
        y += 58
        draw_card(
            pygame.Rect(panel.left + 18, y, side_panel_width - 54, 50),
            "Episode Return",
            f"{self._episode_return:.1f}",
            colors["accent"],
        )
        y += 58
        draw_section_label("Legend", y)
        y += 42

        legend_items = [
            ("Dock", "dock"),
            ("Robot", "robot"),
            ("Dry dirt", "dry"),
            ("Wet dirt", "wet"),
            ("Sticky dirt", "sticky"),
            ("Obstacle", "obstacle"),
            ("Clean", "clean"),
        ]
        legend_col_width = (side_panel_width - 54) // 2
        for index, (label, kind) in enumerate(legend_items):
            col = index % 2
            row = index // 2
            item_x = panel.left + 18 + col * legend_col_width
            item_y = y + row * 34
            icon_rect = pygame.Rect(item_x, item_y - 3, 26, 26)
            draw_legend_marker(icon_rect, kind)
            draw_text(label, (item_x + 36, item_y + 2), colors["muted"])

        if show_home:
            ticks = pygame.time.get_ticks()
            overlay = pygame.Surface((width, height), pygame.SRCALPHA)
            overlay.fill((15, 23, 42, 172))
            surface.blit(overlay, (0, 0))
            home = pygame.Rect(0, 0, 530, 300)
            home.center = (width // 2, height // 2)
            draw_shadow(home, radius=24, alpha=45)
            pygame.draw.rect(surface, (248, 250, 252), home, border_radius=22)
            pygame.draw.rect(surface, colors["panel_line"], home, width=1, border_radius=22)

            logo_center = (home.left + 96, home.top + 92)
            pulse = 1.0 + 0.08 * np.sin(ticks / 180.0)
            logo_rect = pygame.Rect(0, 0, int(88 * pulse), int(88 * pulse))
            logo_rect.center = logo_center
            pygame.draw.circle(surface, (219, 234, 254), logo_center, logo_rect.width // 2)
            pygame.draw.circle(surface, colors["robot"], logo_center, max(24, logo_rect.width // 3))
            pygame.draw.circle(surface, (255, 255, 255), logo_center, max(8, logo_rect.width // 9))
            pygame.draw.circle(surface, colors["robot_dark"], logo_center, max(24, logo_rect.width // 3), width=4)
            for angle in range(0, 360, 45):
                rad = np.deg2rad(angle + ticks / 20.0)
                dot = (
                    int(logo_center[0] + np.cos(rad) * 62),
                    int(logo_center[1] + np.sin(rad) * 62),
                )
                pygame.draw.circle(surface, colors["accent"], dot, 3)

            draw_text("Vacuum Cleaner Simulation", (home.left + 170, home.top + 62), colors["text"], self._title_font)
            draw_text("Loading policy rollout...", (home.left + 172, home.top + 96), colors["muted"], self._font)
            draw_pill(pygame.Rect(home.left + 172, home.top + 130, 118, 26), "STATIC DIRT", colors["accent_soft"], colors["accent"])

            bar = pygame.Rect(home.left + 170, home.top + 182, 300, 16)
            pygame.draw.rect(surface, (226, 232, 240), bar, border_radius=8)
            wave = 0.5 + 0.5 * np.sin(ticks / 260.0)
            fill = bar.copy()
            fill.width = max(18, int(bar.width * (0.35 + 0.55 * wave)))
            pygame.draw.rect(surface, colors["accent"], fill, border_radius=8)

            for i in range(18):
                px = home.left + 48 + ((i * 43 + ticks // 14) % (home.width - 96))
                py = home.bottom - 48 - int(18 * np.sin((ticks / 210.0) + i))
                pygame.draw.circle(surface, (148, 163, 184), (px, py), 2 + (i % 3))

        return np.transpose(pygame.surfarray.array3d(surface), axes=(1, 0, 2))

    def render(self):
        if self.render_mode is None:
            return None
        if self.render_mode == "ansi":
            return self._render_ansi()

        if self.render_mode == "human":
            pygame = self._load_pygame()
            show_home = self._step_count == 0 and self._last_action_name is None
            final_frame = self._render_frame(
                robot_progress=1.0,
                ui_progress=1.0,
                boom_phase=1.0,
                collision_phase=1.0,
                show_home=show_home,
            )
            if self._window is None:
                self._window = pygame.display.set_mode((final_frame.shape[1], final_frame.shape[0]))
                pygame.display.set_caption("VacuumCleaner")
            if self._clock is None:
                self._clock = pygame.time.Clock()

            animate_move = (
                self._previous_robot_pos is not None
                and self._current_state is not None
                and self._previous_robot_pos != self._current_state.robot_pos
                and not self._last_manual_reset
            )
            animate_ui = self._previous_battery is not None or self._previous_step_count is not None
            if show_home:
                progress_values = tuple(i / 36 for i in range(1, 37))
            elif self._last_manual_reset:
                progress_values = tuple(i / 18 for i in range(1, 19))
            elif self._last_move_case == "bump":
                progress_values = tuple(i / 14 for i in range(1, 15))
            elif animate_move or animate_ui:
                progress_values = (0.18, 0.36, 0.58, 0.78, 1.0)
            else:
                progress_values = (1.0,)
            for progress in progress_values:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.close()
                        return None
                if show_home:
                    frame = self._render_frame(
                        robot_progress=1.0,
                        ui_progress=1.0,
                        boom_phase=1.0,
                        collision_phase=1.0,
                        show_home=True,
                    )
                elif progress >= 0.999:
                    frame = final_frame
                else:
                    frame = self._render_frame(
                        robot_progress=progress,
                        ui_progress=progress,
                        boom_phase=progress,
                        collision_phase=progress,
                        show_home=False,
                    )
                surface = pygame.surfarray.make_surface(np.transpose(frame, axes=(1, 0, 2)))
                self._window.blit(surface, (0, 0))
                pygame.display.flip()
                self._clock.tick(30 if show_home or self._last_manual_reset or self._last_move_case == "bump" else max(24, self.metadata["render_fps"]))
            return None
        if self.render_mode == "rgb_array":
            return self._render_frame()
        return None

    def close(self) -> None:
        if self._pygame is not None:
            if self._window is not None:
                self._pygame.display.quit()
            self._pygame.quit()
        self._window = None
        self._clock = None
        self._font = None
        self._small_font = None
        self._title_font = None
        self._mono_font = None
        self._pygame = None


ensure_env_registered()
