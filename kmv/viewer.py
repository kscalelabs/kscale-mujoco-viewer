"""Utilities for rendering the environment."""

import logging
from pathlib import Path
from types import TracebackType
from typing import Sequence

import mujoco
import mujoco.viewer
import numpy as np

from kmv.utils.markers import TrackingConfig, TrackingMarker
from kmv.utils.plotting import Plotter, ThreadedPlotter
from kmv.utils.saving import save_video
from kmv.utils.transforms import rotation_matrix_from_direction
from kmv.utils.types import CommandValue, ModelCache
from kmv.utils.video_writer import StreamingVideoWriter

logger = logging.getLogger(__name__)


class MujocoViewerHandler:
    def __init__(
        self,
        handle: mujoco.viewer.Handle,
        capture_pixels: bool = False,
        video_save_path: str | Path | None = None,
        render_width: int = 640,
        render_height: int = 480,
        make_plots: bool = False,
    ) -> None:
        # breakpoint()
        self.handle = handle
        self._markers: list[TrackingMarker] = []
        self._frames: list[np.ndarray] = []
        self._capture_pixels = capture_pixels
        self._save_path = Path(video_save_path) if video_save_path is not None else None
        self._renderer = None
        self._model_cache = ModelCache.create(self.handle.m)
        self._initial_z_offset: float | None = None
        self._video_writer: StreamingVideoWriter | None = None
        
        # Default video settings
        self._show_frame_number = True
        self._show_sim_time = True
        self._video_quality = 8
        self._fps = 30
        
        self.current_sim_time = 0.0
        self.prev_sim_time = 0.0
        self._total_sim_time_offset = 0.0
        self._total_current_sim_time = 0.0
        self._render_width = render_width
        self._render_height = render_height

        # Initialize real-time plots if requested
        self._make_plots = make_plots
        self._plotter = None
        self._start_time = None
        self._plot_names = set()

        if self._make_plots:
            logger.info("Initializing threaded plotter")
            # Create plotter with appropriate title in a separate thread
            self._plotter = ThreadedPlotter(window_title="MuJoCo Robot Data Plots")
            self._plotter.start()

        if (self._capture_pixels and self.handle.m is not None) or (self._save_path is not None):
            self._renderer = mujoco.Renderer(self.handle.m, width=render_width, height=render_height)
            
            # Initialize video writer with default settings if save path is provided
            if self._save_path is not None:
                pass
                self._init_video_writer()

    def _init_video_writer(self) -> None:
        """Initialize the video writer with current settings."""
        if self._save_path is None:
            return
            
        self._video_writer = StreamingVideoWriter(
            save_path=self._save_path,
            fps=self._fps,
            frame_width=self._render_width,
            frame_height=self._render_height,
            show_frame_number=self._show_frame_number,
            show_sim_time=self._show_sim_time,
            quality=self._video_quality,
        )
        logger.info(f"Initialized video writer for {self._save_path}")

    def setup_video(
        self, 
        show_frame_number: bool = True,
        show_sim_time: bool = True,
        video_quality: int = 8,
        fps: int = 30,
    ) -> None:
        """Configure video recording options.
        
        Args:
            show_frame_number: Whether to show frame numbers on the video
            show_sim_time: Whether to show simulation time on the video
            video_quality: Video quality setting (0-10)
            fps: Frames per second for the video
        """
        # Only close existing writer if it has frames (avoid closing an unused writer)
        if self._video_writer is not None:
            if self._video_writer.frame_count > 0:
                # If we've captured frames, close the writer
                self._video_writer.close()
                self._video_writer = None
            else:
                # If no frames captured yet, just update the writer settings
                self._video_writer.show_frame_number = show_frame_number
                self._video_writer.show_sim_time = show_sim_time
                self._video_writer.fps = fps
                self._video_writer.quality = video_quality
                
                # Update internal settings too
                self._show_frame_number = show_frame_number
                self._show_sim_time = show_sim_time
                self._video_quality = video_quality
                self._fps = fps
                return
        
        # Update settings
        self._show_frame_number = show_frame_number
        self._show_sim_time = show_sim_time
        self._video_quality = video_quality
        self._fps = fps
        
        # Re-initialize the writer with new settings
        if self._save_path is not None:
            self._init_video_writer()

    def setup_camera(
        self,
        render_distance: float = 5.0,
        render_azimuth: float = 90.0,
        render_elevation: float = -30.0,
        render_lookat: list[float] = [0.0, 0.0, 0.5],
        render_track_body_id: int | None = None,
    ) -> None:
        """Setup the camera with the given configuration.

        Args:
            render_distance: Distance from the camera to the target
            render_azimuth: Azimuth angle of the camera
            render_elevation: Elevation angle of the camera
            render_lookat: Lookat position of the camera
            render_track_body_id: Body ID to track with the camera
        """
        self.handle.cam.distance = render_distance
        self.handle.cam.azimuth = render_azimuth
        self.handle.cam.elevation = render_elevation
        self.handle.cam.lookat[:] = render_lookat

        if render_track_body_id is not None:
            self.handle.cam.trackbodyid = render_track_body_id
            self.handle.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING

    def add_plot_group(
        self,
        title: str,
        index_mapping: dict[int, str] | None = None,
        y_axis_min: float | None = None,
        y_axis_max: float | None = None,
    ) -> None:
        """Add a plot group to the viewer."""
        if self._plotter is None:
            raise ValueError("Plotter not initialized. Call `make_plots=True` when initializing the viewer.")
        self._plotter.add_plot_group(title, index_mapping, y_axis_min, y_axis_max)

    def update_plot_group(self, title: str, y_values: list[float], x_value: float = None) -> None:
        """Update a plot group with new data.
        
        Args:
            title: Name of the plot group
            y_values: Y-axis values to update
            x_value: X-axis value, defaults to current simulation time if None
        """
        if self._plotter is None:
            raise ValueError("Plotter not initialized. Call `make_plots=True` when initializing the viewer.")
        
        # Use current sim time if x is not provided
        if x_value is None:
            x_value = self._total_current_sim_time
            
        # Check if using ThreadedPlotter or regular Plotter and adapt parameter order
        if isinstance(self._plotter, ThreadedPlotter):
            # ThreadedPlotter's update_plot_group accepts (group_name, y_values, x_value)
            self._plotter.update_plot_group(title, y_values, x_value)
        else:
            # Regular Plotter's update_plot_group accepts (group_name, x_value, y_values)
            self._plotter.update_plot_group(title, x_value, y_values)

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
        base_pos: tuple[float, float, float] = (0, 0, 1.7),
        scale: float = 0.1,
        rgba: tuple[float, float, float, float] = (0, 1.0, 0, 1.0),
        direction: Sequence[float] | None = None,
        label: str | None = None,
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
        # # Force a sync to ensure the current state is displayed
        # self.handle.sync()

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

    def update_time(self) -> None:
        """Update the time of the viewer."""
        self._current_sim_time = self.handle.d.time
        if self._current_sim_time < self.prev_sim_time:
            self._total_sim_time_offset += self.prev_sim_time
        self._total_current_sim_time = self._current_sim_time + self._total_sim_time_offset
        self.prev_sim_time = self._current_sim_time

    def update_and_sync(self) -> None:
        # Update simulation state
        self.update_time()

        # Ensure forward dynamics are calculated for accurate visualization
        # This is critical for getting correct state when paused or during manual interactions
        if self.handle.m is not None and self.handle.d is not None:
            mujoco.mj_forward(self.handle.m, self.handle.d)

        # Force update the scene with markers
        self._update_scene_markers()

        # If we're capturing, read pixels for video
        if self._renderer is not None and self._video_writer is not None and self._save_path is not None:
            pixels = self.read_pixels()
            self._video_writer.add_frame(pixels, sim_time=self._total_current_sim_time)
            


        # We don't need to render the plotter frame here
        # since it's running in its own thread
        
        # Ensure that the plotter is synchronized with current state
        if self._make_plots and self._plotter is not None and isinstance(self._plotter, ThreadedPlotter):
            flush_success = self._plotter.flush_updates(timeout=0.05)
            if not flush_success:
                logger.warning("Plotter updates not processed in time, plotting thread may be overloaded")

        # Sync the mujoco viewer
        self.sync()
        
        # Clear markers after syncing to avoid accumulation
        self.clear_markers()

    def close(self) -> None:
        """Close the viewer and release resources."""
        if self._video_writer is not None:
            self._video_writer.close()
            self._video_writer = None
            logger.info(f"Saved video to {self._save_path}")

        if self._plotter is not None:
            logger.info("Closing threaded plotter")
            self._plotter.close()
            self._plotter = None

    def add_plot(
        self,
        plot_name: str,
        x_label: str = "Total Sim Time",
        y_label: str = "y",
        y_axis_min: float | None = None,
        y_axis_max: float | None = None,
        group: str | None = None,
    ) -> None:
        """Add a plot to the viewer.

        Args:
            plot_name: Name of the plot
            x_label: Label for the x-axis
            y_label: Label for the y-axis
            y_axis_min: Minimum y-axis value
            y_axis_max: Maximum y-axis value
            group: Group to add the plot to
        """
        if not self._make_plots or self._plotter is None:
            return
            
        self._plotter.add_plot(
            plot_name=plot_name,
            x_label=x_label,
            y_label=y_label,
            y_axis_min=y_axis_min,
            y_axis_max=y_axis_max,
            group=group
        )
        
        # Keep track of plot names
        self._plot_names.add(plot_name)
    
    def update_plot(self, plot_name: str, y_value: float, x_value: float = None) -> None:
        """Update a plot with new data.

        Args:
            plot_name: Name of the plot
            y_value: Y-axis value
            x_value: X-axis value, defaults to current simulation time if None
        """
        if not self._make_plots or self._plotter is None or plot_name not in self._plot_names:
            return
        
        # Use current sim time if x is not provided
        if x_value is None:
            x_value = self._total_current_sim_time
            
        # Check if using ThreadedPlotter or regular Plotter and adapt parameter order
        if isinstance(self._plotter, ThreadedPlotter):
            # ThreadedPlotter's update_plot accepts (plot_name, y, x)
            self._plotter.update_plot(plot_name, y_value, x_value)
        else:
            # Regular Plotter's update_plot accepts (plot_name, x, y)
            self._plotter.update_plot(plot_name, x_value, y_value)


class MujocoViewerHandlerContext:
    def __init__(
        self,
        handle: mujoco.viewer.Handle,
        capture_pixels: bool = False,
        save_path: str | Path | None = None,
        render_width: int = 640,
        render_height: int = 480,
        ctrl_dt: float | None = None,
        make_plots: bool = False,
    ) -> None:
        self.handle = handle
        self.capture_pixels = capture_pixels
        self.save_path = save_path
        self.handler: MujocoViewerHandler | None = None
        self.make_plots = make_plots

        # Options for the renderer.
        self.render_width = render_width
        self.render_height = render_height
        self.ctrl_dt = ctrl_dt

    def __enter__(self) -> MujocoViewerHandler:
        self.handler = MujocoViewerHandler(
            self.handle,
            capture_pixels=self.capture_pixels,
            video_save_path=self.save_path,
            render_width=self.render_width,
            render_height=self.render_height,
            make_plots=self.make_plots,
        )
        
        # Update fps if we have a control timestep
        if self.handler._video_writer is not None and self.ctrl_dt is not None:
            fps = round(1 / float(self.ctrl_dt))
            self.handler.setup_video(fps=fps)
            
        return self.handler

    def __exit__(self, exc_type: type | None, exc_value: Exception | None, traceback: TracebackType | None) -> None:
        logger.info("MujocoViewerHandlerContext.__exit__ called")
        
        # Ensure handler is closed if it exists
        if self.handler is not None:
            logger.info("Closing handler in __exit__")
            self.handler.close()
        
        # If we have a handler and a save path but no streaming writer was used,
        # use the legacy approach to save the video
        if (
            self.handler is not None 
            and self.save_path is not None 
            and self.handler._video_writer is None 
            and self.handler._frames
        ):
            fps = 30
            if self.ctrl_dt is not None:
                fps = round(1 / float(self.ctrl_dt))
            logger.info(f"Using legacy video saving with {len(self.handler._frames)} frames")
            save_video(self.handler._frames, self.save_path, fps=fps)

        # Always close the handle
        logger.info("Closing viewer handle")
        self.handle.close()
        logger.info("MujocoViewerHandlerContext.__exit__ complete")


def launch_passive(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    show_left_ui: bool = False,
    show_right_ui: bool = False,
    capture_pixels: bool = False,
    save_path: str | Path | None = None,
    render_width: int = 640,
    render_height: int = 480,
    ctrl_dt: float | None = None,
    make_plots: bool = False,
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
        render_width: The width of the rendered image
        render_height: The height of the rendered image
        ctrl_dt: The control time step (used to calculate fps)
        make_plots: Whether to show a separate plotting window
        
    Returns:
        A context manager that handles the MujocoViewer lifecycle
    """
    return MujocoViewerHandlerContext(
        mujoco.viewer.launch_passive(
            model=model,
            data=data,
            show_left_ui=show_left_ui,
            show_right_ui=show_right_ui,
        ),
        capture_pixels=capture_pixels,
        save_path=save_path,
        render_width=render_width,
        render_height=render_height,
        ctrl_dt=ctrl_dt,
        make_plots=make_plots,
    )
