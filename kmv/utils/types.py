"""Utility functions for type checking and configuration handling."""

from typing import Optional, TypeVar, cast

from omegaconf import DictConfig

# Define T as TypeVar that allows None to handle Optional values properly
T = TypeVar("T", bound=Optional[object])


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
