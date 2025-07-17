"""Defines types and dataclasses for the viewer."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from enum import Enum, auto
from typing import Literal, Mapping, Optional, Tuple

import mujoco
import numpy as np


@dataclass(frozen=True, slots=True)
class Frame:
    """Single MuJoCo state sample."""

    qpos: np.ndarray
    qvel: np.ndarray
    xfrc_applied: np.ndarray | None = None


Scalars = Mapping[str, float]


@dataclass(frozen=True)
class Msg:
    """Base message class for control-pipe messages."""

    pass


@dataclass(frozen=True)
class ForcePacket(Msg):
    """Array for mouse interaction for xrfc pushes in the GUI."""

    forces: np.ndarray


@dataclass(frozen=True)
class TelemetryPacket(Msg):
    """Key-value rows for the stats table."""

    rows: Mapping[str, float]


@dataclass(frozen=True)
class PlotPacket(Msg):
    """Batch of scalar curves to append to a plot group."""

    group: str
    scalars: Mapping[str, float]


RGBA = Tuple[float, float, float, float]


class GeomType(Enum):
    """Supported MuJoCo primitive shapes for debug markers."""

    SPHERE = auto()
    CAPSULE = auto()
    CYLINDER = auto()
    ELLIPSOID = auto()
    BOX = auto()
    ARROW = auto()
    MESH = auto()

    def to_mj_geom(self) -> mujoco.mjtGeom:
        """Return the matching `mjtGeom` enum value for MuJoCo."""
        return _MJ_MAP[self]


_MJ_MAP: dict[GeomType, mujoco.mjtGeom] = {
    GeomType.SPHERE: mujoco.mjtGeom.mjGEOM_SPHERE,
    GeomType.CAPSULE: mujoco.mjtGeom.mjGEOM_CAPSULE,
    GeomType.CYLINDER: mujoco.mjtGeom.mjGEOM_CYLINDER,
    GeomType.ELLIPSOID: mujoco.mjtGeom.mjGEOM_ELLIPSOID,
    GeomType.BOX: mujoco.mjtGeom.mjGEOM_BOX,
    GeomType.ARROW: mujoco.mjtGeom.mjGEOM_ARROW,
    GeomType.MESH: mujoco.mjtGeom.mjGEOM_MESH,
}


@dataclass(slots=True)
class Marker:
    """Generic debug marker – choose any supported `GeomType`."""

    id: str | int
    pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    geom_type: GeomType = GeomType.SPHERE
    size: Tuple[float, float, float] = (0.05, 0.05, 0.05)
    rgba: RGBA = (1.0, 0.0, 0.0, 1.0)
    body_id: int | None = None
    geom_id: int | None = None
    local_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    orient: Tuple[float, ...] = (
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
    )

    def clone_with(self, **kwargs: object) -> "Marker":
        """Return a new Marker with *kwargs* overwritten."""
        return replace(self, **kwargs)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class _MarkerCmd(Msg, ABC):
    """Base class for all marker commands."""

    id: str | int

    @abstractmethod
    def apply(self, registry: dict[str | int, "Marker"]) -> None: ...


@dataclass(frozen=True, slots=True)
class AddMarker(_MarkerCmd):
    """Add a brand-new marker."""

    marker: Marker

    def apply(self, registry: dict[str | int, "Marker"]) -> None:
        registry.setdefault(self.id, self.marker)


@dataclass(frozen=True, slots=True)
class UpdateMarker(_MarkerCmd):
    """Update existing marker fields **in-place**."""

    fields: Mapping[str, object]

    def apply(self, registry: dict[str | int, "Marker"]) -> None:
        if self.id in registry:
            registry[self.id] = registry[self.id].clone_with(**self.fields)


@dataclass(frozen=True, slots=True)
class RemoveMarker(_MarkerCmd):
    """Delete a marker by id."""

    def apply(self, registry: dict[str | int, "Marker"]) -> None:
        registry.pop(self.id, None)


RenderMode = Literal["window", "offscreen"]


@dataclass(frozen=True, slots=True)
class _TrailCmd(Msg):
    """Base class for all trail commands."""

    id: str | int


@dataclass(frozen=True, slots=True)
class AddTrail(_TrailCmd):
    """Create a brand-new trail with drawing parameters."""

    max_len: int | None = 150
    radius: float = 0.01
    rgba: RGBA = (0.1, 0.6, 1.0, 0.9)
    track_body_id: int | None = None
    track_geom_id: int | None = None
    min_segment_dist: float = 1e-3


@dataclass(frozen=True, slots=True)
class PushTrailPoint(_TrailCmd):
    """Append a single XYZ point to an existing trail."""

    point: Tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class RemoveTrail(_TrailCmd):
    """Delete the trail and all of its segments."""


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
