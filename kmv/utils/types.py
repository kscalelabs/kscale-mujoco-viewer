"""Utility functions for type checking and configuration handling."""

from typing import Optional, Protocol, TypeVar, cast

import mujoco
from attrs import define
from mujoco import mjx
from omegaconf import DictConfig

from kmv.utils.mujoco_helpers import get_map_body_name_to_idx, get_map_geom_name_to_idx

# Define T as TypeVar that allows None to handle Optional values properly
T = TypeVar("T", bound=Optional[object])


class CommandValue(Protocol):
    """Protocol for command values."""

    @property
    def shape(self) -> tuple[int, ...]: ...
    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> float: ...


def get_config_value(
    config: "DictConfig | dict[str, object] | None", key: str, default: Optional[T] = None
) -> Optional[T]:
    """Get a value from config object regardless of its actual type.

    Tries attribute access first (for DictConfig), then falls back to dictionary access.

    Args:
        config: The configuration object
        key: The key to access
        default: Default value to return if key is not found

    Returns:
        The value at the given key or the default
    """
    if config is None:
        return default

    try:
        # Cast the result to match the expected return type
        return cast(Optional[T], getattr(config, key))
    except AttributeError:
        try:
            # Cast the result to match the expected return type
            return cast(Optional[T], config[key])
        except (KeyError, TypeError):
            return default


@define
class ModelCache:
    body_mapping: dict[str, int]
    geom_mapping: dict[str, int]

    @classmethod
    def create(cls, model: mujoco.MjModel | mjx.Model) -> "ModelCache":
        return cls(body_mapping=get_map_body_name_to_idx(model), geom_mapping=get_map_geom_name_to_idx(model))
