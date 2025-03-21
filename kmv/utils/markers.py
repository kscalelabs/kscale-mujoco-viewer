"""Classes for creating and updating markers in a MuJoCo scene."""

import mujoco
import numpy as np
from attrs import define, field

from kmv.utils.mujoco_helpers import get_body_pose_by_name, get_geom_pose_by_name
from kmv.utils.types import ModelCache


def get_target_pose(
    mj_model: mujoco.MjModel, mj_data: mujoco.MjData, target_name: str, target_type: str, cache: ModelCache
) -> tuple[np.ndarray, np.ndarray]:
    if target_type == "body":
        target_pos, target_rot = get_body_pose_by_name(mj_model, mj_data, target_name, cache.body_mapping)
    elif target_type == "geom":
        target_pos, target_rot = get_geom_pose_by_name(mj_model, mj_data, target_name, cache.geom_mapping)
    else:
        raise ValueError(f"Unsupported target type '{target_type}'.")

    if target_pos.shape != (3,):
        raise ValueError(f"Target position has shape {target_pos.shape}, expected (3,)")
    if target_rot.shape != (3, 3):
        raise ValueError(f"Target rotation has shape {target_rot.shape}, expected (3,3)")
    return target_pos, target_rot


@define
class TrackingConfig:
    target_name: str = "world"
    target_type: str = "body"  # "geom" or "body"
    offset: np.ndarray = field(factory=lambda: np.zeros(3))
    track_x: bool = True
    track_y: bool = True
    track_z: bool = True
    track_rotation: bool = True


@define
class Marker:
    name: str
    pos: np.ndarray = field(factory=lambda: np.zeros(3))
    orientation: np.ndarray = field(factory=lambda: np.eye(3))
    color: np.ndarray = field(factory=lambda: np.array([1, 0, 0, 1]))
    scale: np.ndarray = field(factory=lambda: np.array([0.1, 0.1, 0.1]))
    label: str | None = None
    geom: mujoco.MjsGeom | None = None

    def update(self, mj_model: mujoco.MjModel, mj_data: mujoco.MjData) -> None:
        raise NotImplementedError("Update method must be implemented by subclasses.")

    def get_render_data(self) -> dict:
        raise NotImplementedError("Method must be implemented by subclasses.")

    def apply_to_scene(self, scene: mujoco.MjvScene) -> bool:
        """Apply the marker to a MuJoCo scene.

        Args:
            scene: The MjvScene to apply the marker to

        Returns:
            True if the marker was applied successfully, False otherwise
        """
        if scene.ngeom >= scene.maxgeom:
            return False

        # Get the next available geom in the scene
        g = scene.geoms[scene.ngeom]

        # Set basic properties
        g.type = self.geom
        g.size[:] = self.scale
        g.pos[:] = self.pos
        g.mat[:] = self.orientation
        g.rgba[:] = self.color

        # Handle label conversion if needed
        if isinstance(self.label, bytes):
            g.label = self.label
        else:
            g.label = str(self.label).encode("utf-8") if self.label else b""

        # Set other rendering properties
        g.dataid = -1
        g.objtype = mujoco.mjtObj.mjOBJ_UNKNOWN
        g.objid = -1
        g.category = mujoco.mjtCatBit.mjCAT_DECOR
        g.emission = 0
        g.specular = 0.5
        g.shininess = 0.5

        # Increment the geom count
        scene.ngeom += 1
        return True


@define
class TrackingMarker(Marker):
    tracking_cfg: TrackingConfig | None = None
    model_cache: ModelCache | None = None

    def update(self, mj_model: mujoco.MjModel, mj_data: mujoco.MjData) -> None:
        # Use the provided config if available, otherwise self.tracking_cfg.
        config = self.tracking_cfg
        if config is None:
            raise ValueError("Tracking config is not set.")
        if self.model_cache is None:
            raise ValueError("Model cache is not set.")
        target_pos, target_rot = get_target_pose(
            mj_model, mj_data, config.target_name, config.target_type, self.model_cache
        )

        new_pos = self.pos.copy()
        offset = np.array(config.offset, dtype=float)
        if config.track_x:
            new_pos[0] = target_pos[0] + offset[0]
        if config.track_y:
            new_pos[1] = target_pos[1] + offset[1]
        if config.track_z:
            new_pos[2] = target_pos[2] + offset[2]
        self.pos = new_pos

        if config.track_rotation:
            self.orientation = target_rot

    def get_render_data(self) -> dict:
        return {
            "name": self.name,
            "pos": self.pos,
            "orientation": self.orientation,
            "color": self.color,
            "scale": self.scale,
            "geom": self.geom,
        }
