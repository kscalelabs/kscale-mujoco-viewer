"""Thin QOpenGLWindow that asks MuJoCo to draw the current world.

All OpenGL work *must* stay in this thread (Qt requirement).
"""

from PySide6.QtCore import QTimer, Signal
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtOpenGL import QOpenGLWindow
import mujoco


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

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        super().__init__()  # Don't pass parent to QOpenGLWindow
        self.model, self.data = model, data

        self.opt   = mujoco.MjvOption()
        self.scene = mujoco.MjvScene(model, maxgeom=20_000)
        self.cam   = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, self.cam)

        # steady GUI refresh
        self._paint_timer = QTimer(self, interval=16, timeout=self.update)
        self._paint_timer.start()
        self.request_update.connect(self.update)

    # -------- Qt / OpenGL lifecycle ------------------------------------- #
    def initializeGL(self) -> None:
        self.ctx = mujoco.MjrContext(self.model,
                                     mujoco.mjtFontScale.mjFONTSCALE_150)

    def resizeGL(self, w: int, h: int) -> None:
        self._log_w, self._log_h = w, h

    def paintGL(self) -> None:
        dpr = self.devicePixelRatioF()
        rect = mujoco.MjrRect(
            0, 0,
            int(self._log_w * dpr),
            int(self._log_h * dpr),
        )

        mujoco.mjv_updateScene(self.model, self.data,
                               self.opt, None, self.cam,
                               mujoco.mjtCatBit.mjCAT_ALL,
                               self.scene)

        mujoco.mjr_render(rect, self.scene, self.ctx)
