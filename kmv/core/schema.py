# kmv/core/schema.py
"""
Stream-shape declaration used by both parent and GUI processes.

The key idea is to keep *all* shared-memory stream definitions in **one place**
so the two processes never disagree on shapes.

Extend the dict returned by `default_streams()` whenever you need a new bulk
stream (e.g. RGB frames, depth images, proprioceptive sensors…).
"""

from __future__ import annotations

import mujoco
from typing import Mapping, Tuple


def default_streams(model: mujoco.MjModel) -> Mapping[str, Tuple[int, ...]]:
    """
    Return a **mapping** `{stream_name: shape}` describing the bulk data that
    will live in shared memory rings.

    Currently:
        • ``qpos`` : (nq,)
        • ``qvel`` : (nv,)

    Notes
    -----
    *Use **shapes only**, never data-types* – the rings always store ``float64``.
    """
    return {
        "qpos": (model.nq,),
        "qvel": (model.nv,),
        "sim_time": (1,),
        # Example extension:
        # "rgb" : (240, 320, 3),
    }
