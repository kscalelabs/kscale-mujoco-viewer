"""Draw various markers in the viewer."""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Tuple

import mujoco  # ← needed for the mapping

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

    pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    geom_type: GeomType = GeomType.SPHERE
    size: Tuple[float, float, float] = (0.05, 0.05, 0.05)
    rgba: RGBA = (1.0, 0.0, 0.0, 1.0)

    body_id: int | None = None
    geom_id: int | None = None
    local_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)
