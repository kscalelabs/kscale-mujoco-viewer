"""Defines the stream format for the viewer."""

from typing import Mapping, Tuple

import mujoco


def default_streams(model: mujoco.MjModel) -> Mapping[str, Tuple[int, ...]]:
    """Return {stream_name: shape} for the standard viewer."""
    return {
        "qpos": (model.nq,),
        "qvel": (model.nv,),
        "sim_time": (1,),
    }
