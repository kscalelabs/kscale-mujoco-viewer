"""Utilities for rendering the environment."""

from pathlib import Path
from types import TracebackType
from typing import Optional, Sequence, Union

import mujoco
import mujoco.viewer
import numpy as np
from omegaconf import DictConfig

from kmv.utils.saving import save_video
from kmv.utils.transforms import rotation_matrix_from_direction
from kmv.utils.types import get_config_value


class MujocoViewerHandler:
    def __init__(
        self,
        handle: mujoco.viewer.Handle,
        capture_pixels: bool = False,
        save_path: str | Path | None = None,
        config: "DictConfig | dict[str, object] | None" = None,
        render_width: int = 640,
        render_height: int = 480,
    ) -> None:
        self.handle = handle
        self._markers: list[dict[str, object]] = []
        self._frames: list[np.ndarray] = []
        # Initialize renderer for pixel capture if requested
        self._capture_pixels = capture_pixels
        self._save_path = Path(save_path) if save_path is not None else None
        self._render_width = render_width
        self._render_height = render_height
        self._renderer = None
        self._config = config
        self._initial_z_offset: Optional[float] = None  # Store the initial z position + offset
        # If we're going to capture pixels, initialize the renderer now
        if self._capture_pixels and self.handle.m is not None:
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
        pos: Union[list[float], tuple[float, float, float], np.ndarray],
        size: tuple[float, float, float] = (0.1, 0, 0),
        rgba: tuple[float, float, float, float] = (1, 0, 0, 1),
        type: int = mujoco.mjtGeom.mjGEOM_SPHERE,
        mat: Optional[np.ndarray] = None,
        label: str = "",
    ) -> None:
        """Add a marker to be rendered in the scene."""
        self._markers.append(
            {
                "pos": pos,
                "size": size,
                "rgba": rgba,
                "type": type,
                "mat": np.eye(3) if mat is None else mat,
                "label": label if isinstance(label, bytes) else label.encode("utf8") if label else b"",
            }
        )

    def add_commands(self, commands: dict[str, object]) -> None:
        """Add visual representations of commands to the scene."""
        if "linear_velocity_command" in commands:
            command_vel = commands["linear_velocity_command"]

            # Check if it's array-like with indexable values and length
            if (
                hasattr(command_vel, "shape")
                and hasattr(command_vel, "__len__")
                and hasattr(command_vel, "__getitem__")
                and len(command_vel) >= 2
            ):
                try:
                    # Access values safely with type checking
                    x_cmd = float(command_vel[0])
                    y_cmd = float(command_vel[1])

                    # Draw X velocity arrow (forward/backward)
                    self.add_velocity_arrow(
                        x_cmd,
                        base_pos=(0, 0, 1.7),
                        rgba=(1.0, 0.0, 0.0, 0.8),  # Red for X
                        direction=[1.0, 0.0, 0.0],
                        label=f"X Cmd: {x_cmd:.2f}",
                    )

                    # Draw Y velocity arrow (left/right)
                    self.add_velocity_arrow(
                        y_cmd,
                        base_pos=(0, 0, 1.5),
                        rgba=(0.0, 1.0, 0.0, 0.8),  # Green for Y
                        direction=[0.0, 1.0, 0.0],
                        label=f"Y Cmd: {y_cmd:.2f}",
                    )
                except (IndexError, TypeError, ValueError):
                    # Handle errors during access or conversion gracefully
                    pass

    def add_velocity_arrow(
        self,
        command_velocity: float,
        base_pos: tuple[float, float, float] = (0, 0, 1.7),
        scale: float = 0.1,
        rgba: tuple[float, float, float, float] = (0, 1.0, 0, 1.0),
        direction: Optional[Sequence[float]] = None,
        label: Optional[str] = None,
    ) -> None:
        """Add an arrow showing command velocity.

        Args:
            command_velocity: The velocity magnitude
            base_pos: Position for the arrow base
            scale: Scale factor for arrow length
            rgba: Color of the arrow
            direction: Optional direction vector [x,y,z]
            label: Optional text label for the arrow
        """
        # Default to x-axis if no direction provided
        if direction is None:
            direction = [1.0, 0.0, 0.0]

        # For negative velocity, flip the direction
        if command_velocity < 0:
            direction = [-d for d in direction]

        # Get rotation matrix for the direction
        mat = rotation_matrix_from_direction(np.array(direction))

        # Scale the arrow length by the velocity magnitude
        length = abs(command_velocity) * scale

        # Add the arrow marker
        self.add_marker(
            pos=base_pos,
            mat=mat,
            size=(0.02, 0.02, max(0.001, length)),
            rgba=rgba,
            type=mujoco.mjtGeom.mjGEOM_ARROW,
            label=label if label is not None else f"Cmd: {command_velocity:.2f}",
        )

    def _update_scene_markers(self) -> None:
        """Add all current markers to the scene."""
        if self.handle._user_scn is None:
            return

        self._apply_markers_to_scene(self.handle._user_scn)

    def _apply_markers_to_scene(self, scene: mujoco.MjvScene) -> None:
        """Apply markers to the provided scene.

        Args:
            scene: The MjvScene to apply markers to
        """
        for marker in self._markers:
            if scene.ngeom < scene.maxgeom:
                g = scene.geoms[scene.ngeom]

                # Set basic properties
                g.type = marker["type"]
                g.size[:] = marker["size"]
                g.pos[:] = marker["pos"]
                g.mat[:] = marker["mat"]
                g.rgba[:] = marker["rgba"]

                # Handle label conversion if needed
                if isinstance(marker["label"], bytes):
                    g.label = marker["label"]
                else:
                    g.label = str(marker["label"]).encode("utf-8") if marker["label"] else b""

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
        # Force a sync to ensure the current state is displayed
        self.handle.sync()

        # Get the current model and data from the handle
        model = self.handle.m
        data = self.handle.d

        if model is None or data is None:
            # If model or data is not available, return empty array with render dimensions
            return np.zeros((self._render_height, self._render_width, 3), dtype=np.uint8)

        # Initialize or update the renderer if needed
        if self._renderer is None:
            self._renderer = mujoco.Renderer(model, height=self._render_height, width=self._render_width)

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
        render_width: int = 640,
        render_height: int = 480,
        fps: int = 30,
        config: "DictConfig | dict[str, object] | None" = None,
    ) -> None:
        self.handle = handle
        self.capture_pixels = capture_pixels
        self.save_path = save_path
        self.render_width = render_width
        self.render_height = render_height
        self.fps = fps
        self.config = config
        self.handler: Optional[MujocoViewerHandler] = None  # Properly typed

    def __enter__(self) -> MujocoViewerHandler:
        self.handler = MujocoViewerHandler(
            self.handle,
            capture_pixels=self.capture_pixels,
            save_path=self.save_path,
            render_width=self.render_width,
            render_height=self.render_height,
            config=self.config,
        )
        return self.handler

    def __exit__(
        self, exc_type: Optional[type], exc_value: Optional[Exception], traceback: Optional[TracebackType]
    ) -> None:
        # If we have a handler and a save path, save the video before closing
        if self.handler is not None and self.save_path is not None:
            fps = self.fps

            # Get the control timestep if available
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
    render_width: int = 640,
    render_height: int = 480,
    fps: int = 30,
    config: "DictConfig | dict[str, object] | None" = None,
    **kwargs: object,
) -> MujocoViewerHandlerContext:
    """Drop-in replacement for viewer.launch_passive.

    Args:
        model: The MjModel to render
        data: The MjData to render
        show_left_ui: Whether to show the left UI panel
        show_right_ui: Whether to show the right UI panel
        capture_pixels: Whether to capture pixels for video saving
        save_path: Where to save the video (MP4 or GIF)
        render_width: Width of the rendering window
        render_height: Height of the rendering window
        fps: Frames per second for saved video
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
        render_width=render_width,
        render_height=render_height,
        fps=fps,
        config=config,
    )
