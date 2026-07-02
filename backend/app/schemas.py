from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ResetRequest(BaseModel):
    algorithm: str = "ppo"
    seed: int | None = None


class StepRequest(BaseModel):
    algorithm: str = "ppo"
    action: int | str | None = Field(
        default=None,
        description="Discrete action index/name, or omit/use 'auto' to let the selected policy act.",
    )
    mode: Literal["auto", "manual"] | None = None


class AlgorithmInfo(BaseModel):
    id: str
    name: str
    source: str
    model_path: str
    available: bool
    continuous_policy: bool = False
    load_error: str | None = None


class AlgorithmsResponse(BaseModel):
    algorithms: list[AlgorithmInfo]
    actions: list[str]


class EnvStateResponse(BaseModel):
    algorithm: str
    observation: list[float]
    robot_pos: list[int]
    battery: int
    battery_capacity: int
    grid: list[list[int]]
    obstacles: list[list[int]]
    dock: list[int]
    adjacent_cells: list[int]
    action: str | None
    action_index: int | None
    reward: float
    step_count: int
    episode_return: float
    total_dirt: int
    battery_reset: bool
    relocated_to_dock: bool
    terminated: bool
    truncated: bool
    info: dict[str, Any]
