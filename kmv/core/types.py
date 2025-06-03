# kmv/core/types.py
"""
Tiny, zero-dependency dataclasses used across the viewer stack.
No Qt, no multiprocessing imports allowed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Tuple, Optional

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
# 3.  Message types for IPC communication
# -----------------------------------------------------------------------------#

@dataclass
class Msg:
    """Base message type."""
    pass

@dataclass
class ForcePacket(Msg):
    """Force data from GUI interactions."""
    forces: np.ndarray

@dataclass
class TelemetryPacket(Msg):
    """Telemetry data for the stats table."""
    rows: Mapping[str, float]

@dataclass
class PlotPacket(Msg):
    """Plot data for scalar visualizations."""
    group: str
    scalars: Mapping[str, float]


# -----------------------------------------------------------------------------#
# 4.  Public literals
# -----------------------------------------------------------------------------#

RenderMode = Literal["window", "offscreen"]  # kept for backwards compatibility


# -----------------------------------------------------------------------------#
# 5.  Viewer configuration (new)
# -----------------------------------------------------------------------------#

@dataclass(frozen=True, slots=True)
class ViewerConfig:
    # window & widgets
    width: int  = 900
    height: int = 550
    enable_plots: bool = True

    # MuJoCo visual flags
    shadow: bool        = False
    reflection: bool    = False
    contact_force: bool = False
    contact_point: bool = False
    inertia: bool       = False

    # camera (all optional ⇒ keep current free camera)
    camera_distance : Optional[float]                = None
    camera_azimuth  : Optional[float]                = None
    camera_elevation: Optional[float]                = None
    camera_lookat   : Optional[Tuple[float, float, float]] = None
    track_body_id   : Optional[int]                  = None
