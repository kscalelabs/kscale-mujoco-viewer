"""Utilities for rendering the environment."""

import logging
from pathlib import Path
from types import TracebackType
from typing import Optional, Sequence, Tuple

import mujoco
import mujoco.viewer
import numpy as np
from omegaconf import DictConfig

from kmv.utils.markers import TrackingConfig, TrackingMarker
from kmv.utils.saving import save_video
from kmv.utils.transforms import rotation_matrix_from_direction
from kmv.utils.types import CommandValue, ModelCache, get_config_value

logger = logging.getLogger(__name__)


class MujocoViewerHandler:
    def __init__(
        self,
        handle: mujoco.viewer.Handle,
        capture_pixels: bool = False,
        save_path: str | Path | None = None,
        config: "DictConfig | dict[str, object] | None" = None,
    ) -> None:
        self.handle = handle
        self._markers: list[TrackingMarker] = []
        self._frames: list[np.ndarray] = []
        self._capture_pixels = capture_pixels
        self._save_path = Path(save_path) if save_path is not None else None
        self._renderer = None
        self._config = config
        self._model_cache = ModelCache.create(self.handle.m)
        self._initial_z_offset: Optional[float] = None
        if (self._capture_pixels and self.handle.m is not None) or (self._save_path is not None):
            render_width = get_config_value(config, "render_width", 640)
            render_height = get_config_value(config, "render_height", 480)
            self._renderer = mujoco.Renderer(self.handle.m, width=render_width, height=render_height)

    def setup_camera(self, config: "DictConfig | dict[str, object]") -> None:
        """Setup the camera with the given configuration.

        Args:
            config: Configuration with render_distance, render_azimuth, render_elevation,
                   render_lookat, and optionally render_track_body_id.
        """
        self.handle.cam.distance = get_config_value(config, "render_distance", 5.0)
        self.handle.cam.azimuth = get_config_value(config, "render_azimuth", 90.0)
        self.handle.cam.elevation = get_config_value(config, "render_elevation", -30.0)
        self.handle.cam.lookat[:] = get_config_value(config, "render_lookat", [0.0, 0.0, 0.5])

        track_body_id: Optional[int] = get_config_value(config, "render_track_body_id")
        if track_body_id is not None:
            self.handle.cam.trackbodyid = track_body_id
            self.handle.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING

    def copy_data(self, dst: mujoco.MjData, src: mujoco.MjData) -> None:
        """Copy the data from the source to the destination."""
        dst.ctrl[:] = src.ctrl[:]
        dst.act[:] = src.act[:]
        dst.xfrc_applied[:] = src.xfrc_applied[:]
        dst.qpos[:] = src.qpos[:]
        dst.qvel[:] = src.qvel[:]
        dst.time = src.time

    def clear_markers(self) -> None:
        """Clear all markers from the scene."""
        if self.handle._user_scn is not None:
            # Reset the geom counter to effectively clear all markers
            self.handle._user_scn.ngeom = 0
            self._markers = []

    def add_marker(
        self,
        name: str,
        pos: np.ndarray = np.zeros(3),
        orientation: np.ndarray = np.eye(3),
        color: np.ndarray = np.array([1, 0, 0, 1]),
        scale: np.ndarray = np.array([0.1, 0.1, 0.1]),
        label: str | None = None,
        track_geom_name: str | None = None,
        track_body_name: str | None = None,
        track_x: bool = True,
        track_y: bool = True,
        track_z: bool = True,
        track_rotation: bool = True,
        tracking_offset: np.ndarray = np.array([0, 0, 0]),
        geom: int = mujoco.mjtGeom.mjGEOM_SPHERE,
    ) -> None:
        """Add a marker to be rendered in the scene."""
        target_name = "world"
        target_type = "body"
        if track_geom_name is not None:
            target_name = track_geom_name
            target_type = "geom"
        elif track_body_name is not None:
            target_name = track_body_name
            target_type = "body"

        tracking_cfg = TrackingConfig(
            target_name=target_name,
            target_type=target_type,
            offset=tracking_offset,
            track_x=track_x,
            track_y=track_y,
            track_z=track_z,
            track_rotation=track_rotation,
        )
        self._markers.append(
            TrackingMarker(
                name=name,
                pos=pos,
                orientation=orientation,
                color=color,
                scale=scale,
                label=label,
                geom=geom,
                tracking_cfg=tracking_cfg,
                model_cache=self._model_cache,
            )
        )

    def add_commands(self, commands: dict[str, CommandValue]) -> None:
        if "linear_velocity_command" in commands:
            command_vel = commands["linear_velocity_command"]
            if hasattr(command_vel, "shape") and hasattr(command_vel, "__len__") and len(command_vel) >= 2:
                x_cmd = float(command_vel[0])
                y_cmd = float(command_vel[1])
                # Add separate velocity arrows for the x and y commands.
                self.add_velocity_arrow(
                    command_velocity=x_cmd,
                    base_pos=(0, 0, 1.7),
                    scale=0.1,
                    rgba=(1.0, 0.0, 0.0, 0.8),
                    direction=[1.0, 0.0, 0.0],
                    label=f"X: {x_cmd:.2f}",
                )
                self.add_velocity_arrow(
                    command_velocity=y_cmd,
                    base_pos=(0, 0, 1.5),
                    scale=0.1,
                    rgba=(0.0, 1.0, 0.0, 0.8),
                    direction=[0.0, 1.0, 0.0],
                    label=f"Y: {y_cmd:.2f}",
                )

    def add_velocity_arrow(
        self,
        command_velocity: float,
        base_pos: Tuple[float, float, float] = (0, 0, 1.7),
        scale: float = 0.1,
        rgba: Tuple[float, float, float, float] = (0, 1.0, 0, 1.0),
        direction: Optional[Sequence[float]] = None,
        label: Optional[str] = None,
    ) -> None:
        """Convenience method for adding a velocity arrow marker.

        Assumes that velocity arrows track the torso geom (or base body) by default.
        """
        # Default to x-axis if direction not provided.
        if direction is None:
            direction = [1.0, 0.0, 0.0]
        if command_velocity < 0:
            direction = [-d for d in direction]
        mat = rotation_matrix_from_direction(np.array(direction))
        length = abs(command_velocity) * scale

        # Use default tracking: track the torso geometry
        tracking_cfg = TrackingConfig(
            target_name="torso",  # default target name
            target_type="geom",  # default target type
            offset=np.array([0.0, 0.0, 0.5]),
            track_x=True,
            track_y=True,
            track_z=False,  # typically velocity arrows are horizontal
            track_rotation=False,
        )
        marker = TrackingMarker(
            name=label if label is not None else f"Vel: {command_velocity:.2f}",
            pos=np.array(base_pos, dtype=float),
            orientation=mat,
            color=np.array(rgba, dtype=float),
            scale=np.array((0.02, 0.02, max(0.001, length)), dtype=float),
            label=label if label is not None else f"Vel: {command_velocity:.2f}",
            geom=mujoco.mjtGeom.mjGEOM_ARROW,
            tracking_cfg=tracking_cfg,
            model_cache=self._model_cache,
        )
        self._markers.append(marker)

    def _update_scene_markers(self) -> None:
        """Add all current markers to the scene."""
        if self.handle._user_scn is None:
            return

        # Update tracked markers with current positions
        for marker in self._markers:
            marker.update(self.handle.m, self.handle.d)

        # Apply all markers to the scene
        self._apply_markers_to_scene(self.handle._user_scn)

    def add_debug_markers(self) -> None:
        """Add debug markers to the scene using the tracked marker system.

        This adds a sphere at a fixed z height above the robot's base position,
        but following the x,y position of the base.
        """
        if self.handle.d is None:
            return

        # Get the base position from qpos (first 3 values are xyz position)
        base_pos = self.handle.d.qpos[:3].copy()

        # On first call, establish the fixed z height (original z + 0.5)
        if self._initial_z_offset is None:
            self._initial_z_offset = base_pos[2] + 0.5
            print(f"Set fixed z height to: {self._initial_z_offset}")

        # Using the new marker system
        self.add_marker(
            name="debug_marker",
            pos=np.array([base_pos[0], base_pos[1], self._initial_z_offset]),
            scale=np.array([0.1, 0.1, 0.1]),  # Bigger sphere for visibility
            color=np.array([1.0, 0.0, 1.0, 0.8]),  # Magenta color for visibility
            label="Base Pos (fixed z)",
            track_body_name="torso",  # Track the torso body
            track_x=True,
            track_y=True,
            track_z=True,  # Don't track z, keep it fixed
            tracking_offset=np.array([0, 0, 0.5]),  # Offset above the torso
            geom=mujoco.mjtGeom.mjGEOM_ARROW,  # Specify the geom type
        )

        # Print position to console for debugging
        print(f"Marker position: x,y=({base_pos[0]:.2f},{base_pos[1]:.2f}), fixed z={self._initial_z_offset:.2f}")

    def _apply_markers_to_scene(self, scene: mujoco.MjvScene) -> None:
        """Apply markers to the provided scene.

        Args:
            scene: The MjvScene to apply markers to
        """
        for marker in self._markers:
            marker.apply_to_scene(scene)

    def sync(self) -> None:
        """Sync the viewer with current state."""
        self.handle.sync()

    def get_camera(self) -> mujoco.MjvCamera:
        """Get a camera instance configured with current settings."""
        camera = mujoco.MjvCamera()
        camera.type = self.handle.cam.type
        camera.fixedcamid = self.handle.cam.fixedcamid
        camera.trackbodyid = self.handle.cam.trackbodyid
        camera.lookat[:] = self.handle.cam.lookat
        camera.distance = self.handle.cam.distance
        camera.azimuth = self.handle.cam.azimuth
        camera.elevation = self.handle.cam.elevation
        return camera

    def read_pixels(self) -> np.ndarray:
        """Read the current viewport pixels as a numpy array."""
        # Initialize or update the renderer if needed
        if self._renderer is None:
            raise ValueError(
                "Renderer not initialized. "
                "For off-screen rendering, initialize with `capture_pixels=True` or `save_path`"
            )
        # Force a sync to ensure the current state is displayed
        self.handle.sync()

        # Get the current model and data from the handle
        model = self.handle.m
        data = self.handle.d

        if model is None or data is None:
            # If model or data is not available, return empty array with render dimensions
            return np.zeros((self._renderer.height, self._renderer.width, 3), dtype=np.uint8)

        # Get the current camera settings from the viewer
        camera = self.get_camera()

        # Update the scene with the current physics state
        self._renderer.update_scene(data, camera=camera)

        # Add markers to the scene manually
        self._apply_markers_to_scene(self._renderer.scene)

        # Render the scene
        pixels = self._renderer.render()
        return pixels

    def update_and_sync(self) -> None:
        """Update the marks, sync with viewer, and clear the markers."""
        # self.add_debug_markers()
        self._update_scene_markers()
        self.sync()
        if self._save_path is not None:
            self._frames.append(self.read_pixels())
        self.clear_markers()


class MujocoViewerHandlerContext:
    def __init__(
        self,
        handle: mujoco.viewer.Handle,
        capture_pixels: bool = False,
        save_path: str | Path | None = None,
        config: "DictConfig | dict[str, object] | None" = None,
    ) -> None:
        self.handle = handle
        self.capture_pixels = capture_pixels
        self.save_path = save_path
        self.config = config
        self.handler: Optional[MujocoViewerHandler] = None

    def __enter__(self) -> MujocoViewerHandler:
        self.handler = MujocoViewerHandler(
            self.handle,
            capture_pixels=self.capture_pixels,
            save_path=self.save_path,
            config=self.config,
        )
        return self.handler

    def __exit__(
        self, exc_type: Optional[type], exc_value: Optional[Exception], traceback: Optional[TracebackType]
    ) -> None:
        # If we have a handler and a save path, save the video before closing
        if self.handler is not None and self.save_path is not None:
            fps = 30
            ctrl_dt: Optional[float] = get_config_value(self.config, "ctrl_dt")
            if ctrl_dt is not None:
                fps = round(1 / float(ctrl_dt))
            save_video(self.handler._frames, self.save_path, fps=fps)

        # Always close the handle
        self.handle.close()


def launch_passive(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    show_left_ui: bool = False,
    show_right_ui: bool = False,
    capture_pixels: bool = False,
    save_path: str | Path | None = None,
    config: "DictConfig | dict[str, object] | None" = None,
    **kwargs: object,
) -> MujocoViewerHandlerContext:
    """Drop-in replacement for mujoco.viewer.launch_passive.

    See https://github.com/google-deepmind/mujoco/blob/main/python/mujoco/viewer.py

    Args:
        model: The MjModel to render
        data: The MjData to render
        show_left_ui: Whether to show the left UI panel
        show_right_ui: Whether to show the right UI panel
        capture_pixels: Whether to capture pixels for video saving
        save_path: Where to save the video (MP4 or GIF)
        config: Configuration object (supports either DictConfig or standard dict)
        **kwargs: Additional arguments to pass to mujoco.viewer.launch_passive

    Returns:
        A context manager that handles the MujocoViewer lifecycle
    """
    handle = mujoco.viewer.launch_passive(model, data, show_left_ui=show_left_ui, show_right_ui=show_right_ui, **kwargs)
    return MujocoViewerHandlerContext(
        handle,
        capture_pixels=capture_pixels,
        save_path=save_path,
        config=config,
    )
