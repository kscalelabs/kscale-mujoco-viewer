"""Utility functions for type checking and configuration handling."""

from typing import Optional, Protocol, TypeAlias, TypeVar

import mujoco
from attrs import define
from mujoco import mjx

from kmv.utils.mujoco_helpers import get_map_body_name_to_idx, get_map_geom_name_to_idx

# Define T as TypeVar that allows None to handle Optional values properly
T = TypeVar("T", bound=Optional[object])

MujocoModel: TypeAlias = mujoco.MjModel | mjx.Model


class CommandValue(Protocol):
    """Protocol for command values."""

    @property
    def shape(self) -> tuple[int, ...]: ...
    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> float: ...


@define
class ModelCache:
    body_mapping: dict[str, int]
    geom_mapping: dict[str, int]

    @classmethod
    def create(cls, model: MujocoModel) -> "ModelCache":
        return cls(body_mapping=get_map_body_name_to_idx(model), geom_mapping=get_map_geom_name_to_idx(model))
