"""Geometry helpers shared across kmv."""

from typing import Optional

import numpy as np

from kmv.core.types import GeomType, Marker


def orient_z_to_vec(vec: np.ndarray, *, eps: float = 1e-9) -> np.ndarray:
    """Return a 3 × 3 row-major rotation matrix whose **+Z axis** points along *v*.

    This function takes a vector as input, and returns a 3x3 rotation matrix such that
    the +Z axis of that matrix points along (or towards) the vector.
    """
    vec = vec.astype(float, copy=False)

    # Normalize
    v_norm = np.linalg.norm(vec)
    if v_norm < eps:
        return np.eye(3)
    z = vec / v_norm

    # Cross unit Z-axis with vec to get X-axis
    x = np.cross([0.0, 0.0, 1.0], z)
    if np.linalg.norm(x) < eps:  # collinear → pick X-axis
        x = np.array([1.0, 0.0, 0.0])
    x /= np.linalg.norm(x)

    # Cross Z-axis with X-axis to get Y-axis
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)


def capsule_from_to(
    p0: np.ndarray,
    p1: np.ndarray,
    *,
    radius: float,
    seg_id: str | int,
    rgba: tuple[float, float, float, float] = (0.1, 0.6, 1.0, 0.9),
) -> Optional[Marker]:
    """Convenience: return a `Marker` for a capsule whose axis runs from p0 to p1.

    Keeps the +Z-axis convention and fills in `size` & `orient`.

    Returns:
        Marker or None if the segment is too short.
    """
    mid = 0.5 * (p0 + p1)
    d = p1 - p0
    length = float(np.linalg.norm(d))

    # Degenerate segment – nothing to draw.
    if length < 1e-9:
        return None

    rot = orient_z_to_vec(d).reshape(-1)  # row-major
    return Marker(
        id=seg_id,
        pos=tuple(mid),
        geom_type=GeomType.CAPSULE,
        size=(radius, 0.5 * length, radius),
        rgba=rgba,
        orient=tuple(rot),
    )
