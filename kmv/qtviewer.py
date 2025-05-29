"""MuJoCo viewer implementation using Qt/PySide6 with ksim compatibility."""

from __future__ import annotations
import sys
import logging
from pathlib import Path
from typing import Callable, Literal, get_args

import mujoco
import numpy as np
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget
from PySide6.QtCore import QTimer, Qt, QEventLoop
from PySide6.QtGui import QAction

from .renderer import GLViewport
from .engine import SimEngine
from .plots import QPosPlot

logger = logging.getLogger(__name__)

RenderMode = Literal["window", "offscreen"]
Callback = Callable[[mujoco.MjModel, mujoco.MjData, mujoco.MjvScene], None]


class QtViewer(QMainWindow):
    """Qt-based MuJoCo viewer with ksim compatibility.
    
    This viewer provides the same interface as ksim's GlfwMujocoViewer but uses
    Qt/PySide6 for better integration and extensibility.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData | None = None,
        mode: RenderMode = "window",
        title: str = "KMV MuJoCo Viewer",
        width: int | None = None,
        height: int | None = None,
        shadow: bool = False,
        reflection: bool = False,
        contact_force: bool = False,
        contact_point: bool = False,
        inertia: bool = False,
        max_geom: int = 10000,
    ) -> None:
        """Initialize the KMV MuJoCo viewer.

        Args:
            model: MuJoCo model
            data: MuJoCo data (created automatically if None)
            mode: Rendering mode ("window" or "offscreen")
            title: Window title
            width: Window width
            height: Window height
            shadow: Whether to render shadows
            reflection: Whether to render reflections
            contact_force: Whether to render contact forces
            contact_point: Whether to render contact points
            inertia: Whether to render inertia
            max_geom: Maximum number of geometries to render
        """
        # Initialize Qt application if needed
        self.app = QApplication.instance() or QApplication(sys.argv)
        
        super().__init__()
        
        self.setWindowTitle(title)
        self._mode = mode
        self._is_alive = True
        
        # Store MuJoCo objects
        self._model = model
        self._data = data or mujoco.MjData(model)
        
        # Set default dimensions
        if width is None:
            width = 640
        if height is None:
            height = 480
        self.resize(width, height)
        
        # Create the GL viewport (main renderer)
        self._viewport = GLViewport(self._model, self._data, max_geom=max_geom)
        
        # Configure scene options
        self._configure_scene_options(
            shadow=shadow,
            reflection=reflection,
            contact_force=contact_force,
            contact_point=contact_point,
            inertia=inertia,
        )
        
        # Set up the window layout
        if mode == "window":
            self._setup_window_mode()
        else:
            # For offscreen mode, we still create the window but don't show it
            container = QWidget.createWindowContainer(self._viewport, self)
            self.setCentralWidget(container)
    
    def _configure_scene_options(
        self,
        shadow: bool,
        reflection: bool,
        contact_force: bool,
        contact_point: bool,
        inertia: bool,
    ) -> None:
        """Configure the scene rendering options."""
        vopt = self._viewport.opt
        
        # Configure visual options based on parameters
        if shadow:
            vopt.flags[mujoco.mjtVisFlag.mjVIS_SHADOW] = 1
        if reflection:
            vopt.flags[mujoco.mjtVisFlag.mjVIS_REFLECTION] = 1
        if contact_force:
            vopt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = 1
        if contact_point:
            vopt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1
        if inertia:
            vopt.flags[mujoco.mjtVisFlag.mjVIS_INERTIA] = 1
    
    def _setup_window_mode(self) -> None:
        """Set up the viewer for interactive window mode."""
        # Create window container for the GL viewport
        container = QWidget.createWindowContainer(self._viewport, self)
        self.setCentralWidget(container)
        
        # Create toolbar with basic controls
        toolbar = self.addToolBar("Controls")
        
        # Reset action
        reset_action = QAction("Reset", self)
        reset_action.setShortcut("R")
        reset_action.triggered.connect(self._reset_simulation)
        toolbar.addAction(reset_action)
        
        # Show the window
        self.show()
    
    def _reset_simulation(self) -> None:
        """Reset the simulation to initial state."""
        mujoco.mj_resetData(self._model, self._data)
        mujoco.mj_forward(self._model, self._data)
    
    # Properties to match ksim's interface
    @property
    def model(self) -> mujoco.MjModel:
        """Get the MuJoCo model."""
        return self._model
    
    @property
    def data(self) -> mujoco.MjData:
        """Get the MuJoCo data."""
        return self._data
    
    @property
    def cam(self) -> mujoco.MjvCamera:
        """Get the camera object."""
        return self._viewport.cam
    
    @property
    def scn(self) -> mujoco.MjvScene:
        """Get the scene object."""
        return self._viewport.scene
    
    @property
    def vopt(self) -> mujoco.MjvOption:
        """Get the visual options."""
        return self._viewport.opt
    
    @property
    def is_alive(self) -> bool:
        """Check if the viewer is still alive."""
        return self._is_alive and not self.isHidden()
    
    # Main interface methods
    def render(self, callback: Callback | None = None) -> None:
        """Render the current scene.
        
        Args:
            callback: Optional callback function to modify the scene before rendering
        """
        if self._mode == "offscreen":
            raise NotImplementedError("Use 'read_pixels()' for offscreen mode.")
        
        if not self.is_alive:
            return
        
        # Set the callback and trigger a render
        self._viewport.set_callback(callback)
        self._viewport.update()  # Trigger paintGL
        
        # Process events briefly to keep window responsive, but avoid long processing
        # that can cause conflicts with mouse interaction
        self.app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 1)
    
    def read_pixels(self, callback: Callback | None = None) -> np.ndarray:
        """Read pixels from the framebuffer.
        
        Args:
            callback: Optional callback function to modify the scene before rendering
            
        Returns:
            RGB array of shape (height, width, 3)
        """
        # Set the callback
        self._viewport.set_callback(callback)
        
        # Get the pixels from the viewport
        return self._viewport.read_pixels()
    
    def set_camera(self, id: int | str) -> None:
        """Set the camera to use for rendering.
        
        Args:
            id: Camera ID (int) or name (str)
        """
        if isinstance(id, int):
            if id < -1 or id >= self._model.ncam:
                raise ValueError(f"Camera ID {id} is out of range [-1, {self._model.ncam}).")
            
            self.cam.fixedcamid = id
            if id == -1:
                self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                mujoco.mjv_defaultFreeCamera(self._model, self.cam)
            else:
                self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                
        elif isinstance(id, str):
            camera_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, id)
            if camera_id == -1:
                raise ValueError(f'Camera "{id}" does not exist.')
            
            self.cam.fixedcamid = camera_id
            self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        else:
            raise ValueError(f"Invalid camera ID type: {type(id)}")
    
    def close(self) -> None:
        """Close the viewer and clean up resources."""
        self._is_alive = False
        self.hide()
        
        # Clean up Qt resources
        if hasattr(self, '_viewport'):
            self._viewport.close()
        
        super().close()
    
    def closeEvent(self, event) -> None:
        """Handle window close event."""
        self._is_alive = False
        event.accept()

    def set_timer_enabled(self, enabled: bool) -> None:
        """Enable or disable the internal rendering timer.
        
        This can be used when external code (like ksim) is controlling the render loop
        to avoid conflicts between timer-based and manual rendering.
        
        Args:
            enabled: Whether to enable the timer-based rendering
        """
        if enabled:
            if not self._viewport._paint_timer.isActive():
                self._viewport._paint_timer.start()
        else:
            if self._viewport._paint_timer.isActive():
                self._viewport._paint_timer.stop()


# Convenience functions for backward compatibility
def launch_interactive_viewer(model: mujoco.MjModel, **kwargs) -> QtViewer:
    """Launch an interactive MuJoCo viewer window.
    
    Args:
        model: MuJoCo model to visualize
        **kwargs: Additional arguments passed to Viewer
        
    Returns:
        The viewer instance
    """
    return QtViewer(model, mode="window", **kwargs)

