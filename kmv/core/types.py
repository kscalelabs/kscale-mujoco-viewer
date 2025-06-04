"""Defines types and dataclasses for the viewer."""

from dataclasses import dataclass
from typing import Literal, Mapping, Optional, Tuple

import numpy as np


@dataclass(frozen=True, slots=True)
class Frame:
    """Single MuJoCo state sample."""

    qpos: np.ndarray
    qvel: np.ndarray
    xfrc_applied: np.ndarray | None = None


Scalars = Mapping[str, float]


@dataclass
class Msg:
    """Base message class for control-pipe messages."""

    pass


@dataclass
class ForcePacket(Msg):
    """Array for mouse interaction for xrfc pushes in the GUI."""

    forces: np.ndarray


@dataclass
class TelemetryPacket(Msg):
    """Key-value rows for the stats table."""

    rows: Mapping[str, float]


@dataclass
class PlotPacket(Msg):
    """Batch of scalar curves to append to a plot group."""

    group: str
    scalars: Mapping[str, float]


RenderMode = Literal["window", "offscreen"]


@dataclass(frozen=True, slots=True)
class ViewerConfig:
    """Static GUI options sent to the worker at launch time."""

    width: int = 900
    height: int = 550
    enable_plots: bool = True

    shadow: bool = False
    reflection: bool = False
    contact_force: bool = False
    contact_point: bool = False
    inertia: bool = False

    camera_distance: Optional[float] = None
    camera_azimuth: Optional[float] = None
    camera_elevation: Optional[float] = None
    camera_lookat: Optional[Tuple[float, float, float]] = None
    track_body_id: Optional[int] = None
