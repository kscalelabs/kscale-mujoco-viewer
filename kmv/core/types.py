# kmv/core/types.py
"""
Tiny, zero-dependency dataclasses used across the viewer stack.
No Qt, no multiprocessing imports allowed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

import numpy as np


# -----------------------------------------------------------------------------#
# 1.  Physics snapshot – travels through shared memory
# -----------------------------------------------------------------------------#

@dataclass(frozen=True, slots=True)
class Frame:
    """
    One MuJoCo state sample.

    Attributes
    ----------
    qpos : (nq,) float64 array
    qvel : (nv,) float64 array
    xfrc_applied : (nbody, 6) float64 array or None
        Optional external Cartesian forces (used for mouse-drag).
    """
    qpos: np.ndarray
    qvel: np.ndarray
    xfrc_applied: np.ndarray | None = None


# -----------------------------------------------------------------------------#
# 2.  Scalar streams – reward, loss, etc.
# -----------------------------------------------------------------------------#

Scalars = Mapping[str, float]          # alias for duck-typed dict-like objects


# -----------------------------------------------------------------------------#
# 3.  Public literals
# -----------------------------------------------------------------------------#

RenderMode = Literal["window", "offscreen"]  # kept for backwards compatibility
