"""Helper functions for interacting with MuJoCo models and data."""

from typing import Optional, Union

import mujoco
import numpy as np
from mujoco import mjx

PhysicsData = Union[mjx.Data, mujoco.MjData]
PhysicsModel = Union[mjx.Model, mujoco.MjModel]


def get_sensor_data_idxs_by_name(physics_model: PhysicsModel) -> dict[str, tuple[int, int | None]]:
    """Get mappings from sensor names to their data indices."""
    sensor_mappings = {}
    for i in range(len(physics_model.sensor_adr)):
        start = physics_model.sensor_adr[i]
        end = physics_model.sensor_adr[i + 1] if i < len(physics_model.sensor_adr) - 1 else None

        name_start = physics_model.name_sensoradr[i]
        name = bytes(physics_model.names[name_start:]).decode("utf-8").split("\x00")[0]
        sensor_mappings[name] = (start, end)
    return sensor_mappings


def get_qpos_data_idxs_by_name(physics_model: PhysicsModel) -> dict[str, tuple[int, int | None]]:
    """Get mappings from joint names to their position indices."""
    qpos_mappings = {}
    for i in range(len(physics_model.jnt_qposadr)):
        start = physics_model.jnt_qposadr[i]
        end = physics_model.jnt_qposadr[i + 1] if i < len(physics_model.jnt_qposadr) - 1 else None

        name_start = physics_model.name_jntadr[i]
        name = bytes(physics_model.names[name_start:]).decode("utf-8").split("\x00")[0]
        qpos_mappings[name] = (start, end)
    return qpos_mappings


def get_qvelacc_data_idxs_by_name(physics_model: PhysicsModel) -> dict[str, tuple[int, int | None]]:
    """Get mappings from joint names to their velocity/acceleration indices."""
    qvelacc_mappings = {}
    for i in range(len(physics_model.jnt_dofadr)):
        start = physics_model.jnt_dofadr[i]
        end = physics_model.jnt_dofadr[i + 1] if i < len(physics_model.jnt_dofadr) - 1 else None

        name_start = physics_model.name_jntadr[i]
        name = bytes(physics_model.names[name_start:]).decode("utf-8").split("\x00")[0]
        qvelacc_mappings[name] = (start, end)
    return qvelacc_mappings


def get_ctrl_data_idx_by_name(physics_model: PhysicsModel) -> dict[str, int]:
    """Get mappings from actuator names to their control indices."""
    ctrl_mappings = {}
    for i in range(len(physics_model.name_actuatoradr)):
        name_start = physics_model.name_actuatoradr[i]
        name = bytes(physics_model.names[name_start:]).decode("utf-8").split("\x00")[0]
        ctrl_mappings[name] = i
    return ctrl_mappings


def get_map_geom_name_to_idx(physics_model: PhysicsModel) -> dict[str, int]:
    """Get mappings from geometry names to their indices."""
    geom_mappings = {}
    for i in range(physics_model.ngeom):
        name_start = physics_model.name_geomadr[i]
        name = bytes(physics_model.names[name_start:]).decode("utf-8").split("\x00")[0]
        geom_mappings[name] = i
    return geom_mappings


def get_map_body_name_to_idx(physics_model: PhysicsModel) -> dict[str, int]:
    """Get mappings from body names to their indices."""
    body_mappings = {}
    for i in range(physics_model.nbody):
        name_start = physics_model.name_bodyadr[i]
        name = bytes(physics_model.names[name_start:]).decode("utf-8").split("\x00")[0]
        body_mappings[name] = i
    return body_mappings


def get_floor_idx(physics_model: PhysicsModel, floor_name: str = "floor") -> int | None:
    """Get the index of the floor geometry."""
    geom_mappings = get_map_geom_name_to_idx(physics_model)
    assert floor_name in geom_mappings, f"Floor name {floor_name} not found in model"
    return geom_mappings[floor_name]


def is_body_in_contact(
    body_name: str,
    physics_model: PhysicsModel,
    mjx_data: PhysicsData,
) -> bool:
    """Check if a body is in contact."""
    # TODO: implement this properly...
    return False


def get_body_pose(data: PhysicsData, body_idx: int) -> tuple[np.ndarray, np.ndarray]:
    """Get the position and rotation matrix of a body.

    Args:
        data: MuJoCo data object
        body_idx: Index of the body

    Returns:
        tuple of (position, rotation_matrix)
        - position: 3D position vector
        - rotation_matrix: 3x3 rotation matrix
    """
    # Get position
    position = data.xpos[body_idx].copy()

    # Get quaternion and convert to rotation matrix
    quat = data.xquat[body_idx].copy()
    rot_mat = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(rot_mat, quat)
    rot_mat = rot_mat.reshape(3, 3)

    return position, rot_mat


def get_geom_pose(data: PhysicsData, geom_idx: int) -> tuple[np.ndarray, np.ndarray]:
    """Get the position and rotation matrix of a geometry.

    Args:
        data: MuJoCo data object
        geom_idx: Index of the geometry

    Returns:
        tuple of (position, rotation_matrix)
        - position: 3D position vector
        - rotation_matrix: 3x3 rotation matrix
    """
    # Get position
    position = data.geom_xpos[geom_idx].copy()

    # Get rotation matrix (already available, just need to reshape)
    rot_mat = data.geom_xmat[geom_idx].reshape(3, 3).copy()

    return position, rot_mat


def get_body_pose_by_name(
    model: PhysicsModel, data: PhysicsData, body_name: str, body_mapping: Optional[dict[str, int]] = None
) -> tuple[np.ndarray, np.ndarray]:
    if body_mapping is None:
        body_mapping = get_map_body_name_to_idx(model)
    if body_name not in body_mapping:
        raise ValueError(f"Body '{body_name}' not found in model")
    body_idx = body_mapping[body_name]
    return get_body_pose(data, body_idx)


def get_geom_pose_by_name(
    model: PhysicsModel,
    data: PhysicsData,
    geom_name: str,
    geom_mappings: Optional[dict[str, int]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    if geom_mappings is None:
        geom_mappings = get_map_geom_name_to_idx(model)
    if geom_name not in geom_mappings:
        raise ValueError(f"Geometry '{geom_name}' not found in model")

    geom_idx = geom_mappings[geom_name]
    return get_geom_pose(data, geom_idx)
