"""Utility functions for transforming data."""

import numpy as np


def rotation_matrix_from_direction(direction: np.ndarray, reference: np.ndarray = np.array([0, 0, 1])) -> np.ndarray:
    """Compute a rotation matrix that aligns the reference vector with the direction vector."""
    # Normalize direction vector
    dir_vec = np.array(direction, dtype=float)
    norm = np.linalg.norm(dir_vec)
    if norm < 1e-10:  # Avoid division by zero
        return np.eye(3)

    dir_vec = dir_vec / norm

    # Normalize reference vector
    ref_vec = np.array(reference, dtype=float)
    ref_vec = ref_vec / np.linalg.norm(ref_vec)

    # Simple case: vectors are nearly aligned
    if np.abs(np.dot(dir_vec, ref_vec) - 1.0) < 1e-10:
        return np.eye(3)

    # Simple case: vectors are nearly opposite
    if np.abs(np.dot(dir_vec, ref_vec) + 1.0) < 1e-10:
        # Flip around x-axis for [0,0,1] reference
        if np.allclose(ref_vec, [0, 0, 1]):
            return np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        # General case
        else:
            # Find an axis perpendicular to the reference
            perp = np.cross(ref_vec, [1, 0, 0])
            if np.linalg.norm(perp) < 1e-10:
                perp = np.cross(ref_vec, [0, 1, 0])
            perp = perp / np.linalg.norm(perp)

            # Rotate 180 degrees around this perpendicular axis
            c = -1  # cos(π)
            s = 0  # sin(π)
            t = 1 - c
            x, y, z = perp

            return np.array(
                [
                    [t * x * x + c, t * x * y - z * s, t * x * z + y * s],
                    [t * x * y + z * s, t * y * y + c, t * y * z - x * s],
                    [t * x * z - y * s, t * y * z + x * s, t * z * z + c],
                ]
            )

    # General case: use cross product to find rotation axis
    axis = np.cross(ref_vec, dir_vec)
    axis = axis / np.linalg.norm(axis)

    # Angle between vectors
    angle = np.arccos(np.clip(np.dot(ref_vec, dir_vec), -1.0, 1.0))

    # Rodrigues rotation formula
    c = np.cos(angle)
    s = np.sin(angle)
    t = 1 - c
    x, y, z = axis

    return np.array(
        [
            [t * x * x + c, t * x * y - z * s, t * x * z + y * s],
            [t * x * y + z * s, t * y * y + c, t * y * z - x * s],
            [t * x * z - y * s, t * y * z + x * s, t * z * z + c],
        ]
    )
