"""Small public dataclasses shared between producer and viewer."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Mapping

import numpy as np


@dataclass(frozen=True, slots=True)
class Frame:
    """Snapshot of simulator state that the viewer can consume."""
    qpos: np.ndarray
    qvel: np.ndarray
    xfrc_applied: np.ndarray | None = None


Scalars = Mapping[str, float]        # time-series values (reward, loss, …)

RenderMode = Literal["window", "offscreen"]