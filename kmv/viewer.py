"""MuJoCo viewer implementation using Qt/PySide6 with ksim compatibility."""

from __future__ import annotations
import sys
import logging
from pathlib import Path
from typing import Callable, Literal, get_args

import mujoco
import numpy as np
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QLabel, QStatusBar
from PySide6.QtCore import QTimer, Qt, QEventLoop, Signal
from PySide6.QtGui import QAction

from .renderer import GLViewport
from .utils.plotting import PhysicsPlotsDock

logger = logging.getLogger(__name__)

RenderMode = Literal["window", "offscreen"]
Callback = Callable[[mujoco.MjModel, mujoco.MjData, mujoco.MjvScene], None]

class DefaultMujocoViewer:
    """MuJoCo viewer implementation using offscreen OpenGL context."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData | None = None,
        width: int = 320,
        height: int = 240,
        max_geom: int = 10000,
    ) -> None:
        """Initialize the default MuJoCo viewer.

        Args:
            model: MuJoCo model
            data: MuJoCo data
            width: Width of the viewer
            height: Height of the viewer
            max_geom: Maximum number of geoms to render
        """
        super().__init__()

        if data is None:
            data = mujoco.MjData(model)

        self.model = model
        self.data = data
        self.width = width
        self.height = height

        # Validate framebuffer size
        if width > model.vis.global_.offwidth or height > model.vis.global_.offheight:
            raise ValueError(
                f"Image size ({width}x{height}) exceeds offscreen buffer size "
                f"({model.vis.global_.offwidth}x{model.vis.global_.offheight}). "
                "Increase `offwidth`/`offheight` in the XML model."
            )

        # Offscreen rendering context
        self._gl_context = mujoco.gl_context.GLContext(width, height)
        self._gl_context.make_current()

        # MuJoCo scene setup
        self.scn = mujoco.MjvScene(model, maxgeom=max_geom)
        self.vopt = mujoco.MjvOption()
        self.pert = mujoco.MjvPerturb()
        self.rect = mujoco.MjrRect(0, 0, width, height)
        self.cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, self.cam)

        self.ctx = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.ctx)

    def set_camera(self, id: int | str) -> None:
        """Set the camera to use."""
        if isinstance(id, int):
            if id < -1 or id >= self.model.ncam:
                raise ValueError(f"Camera ID {id} is out of range [-1, {self.model.ncam}).")
            # Set up camera
            self.cam.fixedcamid = id
            if id == -1:
                self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                mujoco.mjv_defaultFreeCamera(self.model, self.cam)
            else:
                self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        elif isinstance(id, str):
            camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, id)
            if camera_id == -1:
                raise ValueError(f'The camera "{id}" does not exist.')
            # Set up camera
            self.cam.fixedcamid = camera_id
            self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        else:
            raise ValueError(f"Invalid camera ID: {id}")

    def read_pixels(self, callback: Callback | None = None) -> np.ndarray:
        self._gl_context.make_current()

        # Update scene.
        mujoco.mjv_updateScene(
            self.model,
            self.data,
            self.vopt,
            self.pert,
            self.cam,
            mujoco.mjtCatBit.mjCAT_ALL.value,
            self.scn,
        )

        if callback is not None:
            callback(self.model, self.data, self.scn)

        # Render.
        mujoco.mjr_render(self.rect, self.scn, self.ctx)

        # Read pixels.
        rgb_array = np.empty((self.height, self.width, 3), dtype=np.uint8)
        mujoco.mjr_readPixels(rgb_array, None, self.rect, self.ctx)
        return np.flipud(rgb_array)

    def render(self, callback: Callback | None = None) -> None:
        raise NotImplementedError("Default viewer does not support rendering.")

    def close(self) -> None:
        if self._gl_context:
            self._gl_context.free()
            self._gl_context = None
        if self.ctx:
            self.ctx.free()
            self.ctx = None


class QtViewer(QMainWindow):
    """Qt-based MuJoCo viewer with ksim compatibility.
    
    This viewer provides the same interface as ksim's GlfwMujocoViewer but uses
    Qt/PySide6 for better integration and extensibility.
    """

    # Signal emitted when simulation data is updated
    simulation_step = Signal(float)  # emits simulation time

    DEFAULT_WIN_W = 900
    DEFAULT_WIN_H = 550
    DEFAULT_DOCK_W = 250

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
        show_live_plots: bool = True,
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
            show_live_plots: Whether to show live plotting panels
        """
        # Initialize Qt application if needed
        self.app = QApplication.instance() or QApplication(sys.argv)
        
        super().__init__()
        
        self.setWindowTitle(title)
        self._mode = mode
        self._is_alive = True
        self._show_live_plots = show_live_plots
        
        self._time_offset   = 0.0   # accumulated time from previous episodes
        self._last_mj_time  = 0.0   # last MuJoCo time we saw
        
        # Store MuJoCo objects
        self._model = model
        self._data = data or mujoco.MjData(model)  # keep a handle instead of accessing via `_viewport` only
        
        # Set default dimensions
        if width is None:
            width = self.DEFAULT_WIN_W
        if height is None:
            height = self.DEFAULT_WIN_H
        self.resize(width, height)
        
        # Create the GL viewport (main renderer)
        self._viewport = GLViewport(self._model, self._data, max_geom=max_geom)
        self._viewport.set_mjdata(self._data)
        
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

        # FPS label in status-bar
        self._fps_label = QLabel("FPS: --")
        status = QStatusBar(self)
        status.addPermanentWidget(self._fps_label)
        self.setStatusBar(status)

        self._viewport.fps_changed.connect(  # update label whenever renderer tells us
            lambda v: self._fps_label.setText(f"FPS: {v}")
        )
        
        # Add live plots if requested and in window mode
        if self._show_live_plots and mode == "window":
            self._setup_live_plots()

        # NEW: menus & toolbar
        if mode == "window":
            self._build_menus_and_toolbar()
    
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
    
    def _setup_live_plots(self) -> None:
        """Set up live plotting panels as dock widgets."""
        self._phys_plots = PhysicsPlotsDock(self._model, self._data, parent=self)
        self._phys_plots.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._phys_plots)

        # start hidden
        self._phys_plots.hide()
        self._phys_plots.toggleViewAction().setChecked(False)

        # auto-size when shown
        self._phys_plots.visibilityChanged.connect(self._on_dock_shown)

        # connect data updates
        self.simulation_step.connect(self._phys_plots.on_step)
    
    def _build_menus_and_toolbar(self) -> None:
        """Create View menu + toolbar entries for all dock widgets."""
        menubar = self.menuBar()
        view_menu = menubar.addMenu("&View")

        # Use Qt's built-in toggle action so checked state ↔ visibility stay
        # synchronised automatically.
        if hasattr(self, '_phys_plots'):
            phys_action = self._phys_plots.toggleViewAction()
            phys_action.setText("Physics Plots")
            view_menu.addAction(phys_action)

            # (Optional but nice) duplicate the same action in a toolbar
            toolbar = self.addToolBar("View")
            toolbar.setMovable(False)
            toolbar.addAction(phys_action)
    
    def _reset_simulation(self) -> None:
        """Reset the simulation to initial state."""
        mujoco.mj_resetData(self._model, self._data)
        mujoco.mj_forward(self._model, self._data)
        
        # Reset plot data if live plots are enabled
        if self._show_live_plots and hasattr(self, '_phys_plots'):
            self._phys_plots.reset()
    
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
    def render(self, callback: Callback | None = None, max_time_ms: int = 5) -> None:
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
        # self._viewport.update()  # Trigger paintGL
        
        # Emit simulation step signal for live plots
        if self._show_live_plots:
            cur = float(self._data.time)
            # Detect reset: MuJoCo time went backwards
            if cur < self._last_mj_time - 1e-9:
                self._time_offset += self._last_mj_time
            self._last_mj_time = cur
            total_sim_time = self._time_offset + cur
            self.simulation_step.emit(total_sim_time)
        
        # Give Qt a tiny time-slice (e.g. 5 ms).  This keeps the UI responsive
        # but prevents an infinite loop when continuous input events are present.
        self.app.processEvents(
            QEventLoop.ProcessEventsFlag.AllEvents,
            max_time_ms
        )
        self._viewport.request_update.emit()
    
    def read_pixels(self, callback: Callback | None = None) -> np.ndarray:
        """Read pixels from the framebuffer.
        
        Args:
            callback: Optional callback function to modify the scene before rendering
            
        Returns:
            RGB array of shape (height, width, 3)
        """
        # Set the callback
        self._viewport.set_callback(callback)
        
        # Emit simulation step signal for live plots
        if self._show_live_plots:
            cur = float(self._data.time)
            # Detect reset: MuJoCo time went backwards
            if cur < self._last_mj_time - 1e-9:
                self._time_offset += self._last_mj_time
            self._last_mj_time = cur
            total_sim_time = self._time_offset + cur
            self.simulation_step.emit(total_sim_time)
        
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
        
        # Clean up plot widgets
        if self._show_live_plots and hasattr(self, '_phys_plots'):
            self._phys_plots.close()
        
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

    def _on_dock_shown(self, visible: bool) -> None:
        """Handle dock widget visibility change."""
        if not visible:
            return
        dock_w = min(self.DEFAULT_DOCK_W, self.width() // 3)
        self.resizeDocks([self._phys_plots], [dock_w], Qt.Orientation.Horizontal)


    def set_mjdata(self, data: mujoco.MjData) -> None:
        """
        Replace the MjData object used by the viewer and its sub-widgets.
        Call this immediately after every environment reset.
        """
        self._data = data
        self._viewport.set_mjdata(data)
        if hasattr(self, "_phys_plots"):
            self._phys_plots.set_mjdata(data)



