"""Geometry helpers shared across kmv."""

import numpy as np

from kmv.core.types import GeomType, Marker


def orient_z_to_vec(v: np.ndarray, *, eps: float = 1e-9) -> np.ndarray:
    """Return a 3 × 3 row-major rotation matrix whose **+Z axis** points along *v*.

    Behaviour matches the hand-written version in *examples/default_humanoid.py*.
    """
    v = v.astype(float, copy=False)
    v_norm = np.linalg.norm(v)
    if v_norm < eps:
        return np.eye(3)

    z = v / v_norm
    x = np.cross([0.0, 0.0, 1.0], z)
    if np.linalg.norm(x) < eps:  # collinear → pick X-axis
        x = np.array([1.0, 0.0, 0.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)  # rows: x y z


def capsule_between(
    p0: np.ndarray,
    p1: np.ndarray,
    *,
    radius: float,
    seg_id: str | int,
    rgba: tuple[float, float, float, float] = (0.1, 0.6, 1.0, 0.9),
) -> Marker:
    """Convenience: return a `Marker` for a capsule whose axis runs p0 → p1.

    Keeps the +Z-axis convention and fills in `size` & `orient`.
    """
    mid = 0.5 * (p0 + p1)
    d = p1 - p0
    length = float(np.linalg.norm(d))
    if length < 1e-9:
        return Marker(id=seg_id, pos=tuple(mid))  # degenerate

    rot = orient_z_to_vec(d).reshape(-1)  # row-major
    return Marker(
        id=seg_id,
        pos=tuple(mid),
        geom_type=GeomType.CAPSULE,
        size=(radius, 0.5 * length, radius),
        rgba=rgba,
        orient=tuple(rot),
    )
