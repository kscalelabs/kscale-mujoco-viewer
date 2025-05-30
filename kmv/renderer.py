"""Thin QOpenGLWindow that asks MuJoCo to draw the current world.

All OpenGL work *must* stay in this thread (Qt requirement).
"""

from typing import Callable
import numpy as np
from PySide6.QtCore import QTimer, Signal, Qt
from PySide6.QtGui import QSurfaceFormat, QMouseEvent
from PySide6.QtOpenGL import QOpenGLWindow
from PySide6.QtOpenGLWidgets import QOpenGLWidget
import mujoco
import time


# Configure one global surface format before any GL context exists.
_fmt = QSurfaceFormat()
_fmt.setDepthBufferSize(24)
_fmt.setStencilBufferSize(8)
_fmt.setSamples(4)
_fmt.setSwapInterval(1)           # v-sync
_fmt.setRenderableType(QSurfaceFormat.RenderableType.OpenGL)
QSurfaceFormat.setDefaultFormat(_fmt)


class GLViewport(QOpenGLWindow):
    """Read-only viewer; render loop is ~60 Hz, independent of physics rate."""

    # External widgets can call .request_update.emit() for an immediate repaint
    request_update = Signal()
    fps_changed    = Signal(int)

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, max_geom: int = 20000) -> None:
        super().__init__()  # Don't pass parent to QOpenGLWindow
        self.model, self.data = model, data

        self.opt   = mujoco.MjvOption()
        self.scene = mujoco.MjvScene(model, maxgeom=max_geom)
        self.cam   = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, self.cam)

        # Callback support for ksim compatibility
        self._callback = None

        # Mouse interaction state
        self._mouse_last_x = 0
        self._mouse_last_y = 0
        self._mouse_button_left = False
        self._mouse_button_right = False
        self._mouse_button_middle = False
        
        # Control update frequency during mouse interaction
        self._last_mouse_update = 0

        self._frame_ctr     = 0
        self._last_fps_time = time.time()

        self._paint_timer = QTimer(self)
        self._paint_timer.setInterval(16)  # ~60Hz
        self._paint_timer.setSingleShot(False)
        self._paint_timer.timeout.connect(self.update)
        self._paint_timer.start()
        
        self.request_update.connect(self.update)

    def set_callback(self, callback: Callable[[mujoco.MjModel, mujoco.MjData, mujoco.MjvScene], None] | None) -> None:
        """Set a callback function to be called before rendering.
        
        Args:
            callback: Function that takes (model, data, scene) and can modify the scene
        """
        self._callback = callback

    def initializeGL(self) -> None:
        self.ctx = mujoco.MjrContext(self.model,
                                     mujoco.mjtFontScale.mjFONTSCALE_150)

    def resizeGL(self, w: int, h: int) -> None:
        self._log_w, self._log_h = w, h

    def paintGL(self) -> None:
        self._render_scene()

        self._frame_ctr += 1
        now = time.time()
        if now - self._last_fps_time >= 1.0:
            fps = self._frame_ctr
            self._frame_ctr = 0
            self._last_fps_time = now
            self.fps_changed.emit(fps)

    def _render_scene(self) -> None:
        """Internal method to render the scene with callback support."""
        dpr = self.devicePixelRatioF()
        rect = mujoco.MjrRect(
            0, 0,
            int(self._log_w * dpr),
            int(self._log_h * dpr),
        )

        # Update the scene
        mujoco.mjv_updateScene(self.model, self.data,
                               self.opt, None, self.cam,
                               mujoco.mjtCatBit.mjCAT_ALL,
                               self.scene)

        # Call the callback if provided (for markers, etc.)
        if self._callback is not None:
            try:
                self._callback(self.model, self.data, self.scene)
            except Exception as e:
                print(f"Warning: Callback error: {e}")

        # Render the scene
        mujoco.mjr_render(rect, self.scene, self.ctx)

    def read_pixels(self) -> np.ndarray:
        """Read pixels from the framebuffer.
        
        Returns:
            RGB array of shape (height, width, 3)
        """
        # Make sure we have a valid GL context
        if not self.isValid():
            raise RuntimeError("OpenGL context is not valid")
        
        self.makeCurrent()
        
        # Render the scene to the framebuffer
        self._render_scene()
        
        # Get the framebuffer dimensions
        dpr = self.devicePixelRatioF()
        width = int(self._log_w * dpr)
        height = int(self._log_h * dpr)
        
        # Render pixels to the framebuffer
        rect = mujoco.MjrRect(0, 0, width, height)
        rgb_array = np.empty((height, width, 3), dtype=np.uint8)
        mujoco.mjr_readPixels(rgb_array, None, rect, self.ctx)
        rgb_array = np.flipud(rgb_array)
        
        self.doneCurrent()
        
        return rgb_array

    # Mouse event handlers for camera interaction
    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Handle mouse press events for camera control."""
        self._mouse_last_x = event.position().x()
        self._mouse_last_y = event.position().y()
        
        if event.button() == Qt.MouseButton.LeftButton:
            self._mouse_button_left = True
        elif event.button() == Qt.MouseButton.RightButton:
            self._mouse_button_right = True
        elif event.button() == Qt.MouseButton.MiddleButton:
            self._mouse_button_middle = True
            
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """Handle mouse release events."""
        if event.button() == Qt.MouseButton.LeftButton:
            self._mouse_button_left = False
        elif event.button() == Qt.MouseButton.RightButton:
            self._mouse_button_right = False
        elif event.button() == Qt.MouseButton.MiddleButton:
            self._mouse_button_middle = False
            
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Handle mouse move events for camera control."""
        current_x = event.position().x()
        current_y = event.position().y()
        
        dx = current_x - self._mouse_last_x
        dy = current_y - self._mouse_last_y
        
        # Camera controls similar to standard MuJoCo viewer
        if self._mouse_button_left:
            # Rotate camera
            self.cam.azimuth += dx * 0.5
            self.cam.elevation -= dy * 0.5
            # Clamp elevation
            self.cam.elevation = max(-90, min(90, self.cam.elevation))
            
        elif self._mouse_button_right:
            # Zoom camera
            self.cam.distance *= (1.0 + 0.01 * dy)
            self.cam.distance = max(0.1, min(100.0, self.cam.distance))
            
        elif self._mouse_button_middle:
            # Pan camera (translate lookat point)
            # This is more complex and would require proper coordinate transformations
            # For now, just implement basic panning
            scale = 0.001 * self.cam.distance
            self.cam.lookat[0] += dx * scale
            self.cam.lookat[1] -= dy * scale
        
        self._mouse_last_x = current_x
        self._mouse_last_y = current_y
        
        # Only trigger update if we're actively interacting and some time has passed
        # This reduces conflicts with external render loops
        current_time = time.time()
        if (self._mouse_button_left or self._mouse_button_right or self._mouse_button_middle):
            if current_time - self._last_mouse_update > 0.016:  # ~60fps limit
                self.request_update.emit() 
                self._last_mouse_update = current_time
        
        event.accept()

    def wheelEvent(self, event) -> None:
        """Handle mouse wheel events for zooming."""
        delta = event.angleDelta().y()
        zoom_factor = 1.0 + (delta / 1000.0)
        self.cam.distance *= zoom_factor
        self.cam.distance = max(0.1, min(100.0, self.cam.distance))
        
        # Don't trigger immediate update - let the timer or external render calls handle it
        event.accept()
