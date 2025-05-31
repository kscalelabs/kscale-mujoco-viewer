"""High-level MuJoCo viewer (interactive Qt window or off-screen)."""

from __future__ import annotations
import sys
from typing import Callable

import mujoco
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QStatusBar,
    QWidget,
    QDockWidget,
)

from kmv.ui.gl.viewport import GLViewport
from kmv.ui.plotting.plots import ScalarPlot
from kmv.core.types import RenderMode, Frame
from kmv.core.ring import Ring


Callback = Callable[[mujoco.MjModel, mujoco.MjData, mujoco.MjvScene], None]


class QtViewer(QMainWindow):
    """
    Interactive on-screen viewer.

    Extra keyword-args correspond 1-to-1 with the flags `get_viewer()` passes
    in (shadow, reflection, …) so the RL loop doesn't need to change.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData | None = None,
        *,
        ring: Ring[Frame] | None = None,
        mode: RenderMode = "window",
        width: int = 900,
        height: int = 550,
        shadow: bool = False,
        reflection: bool = False,
        contact_force: bool = False,
        contact_point: bool = False,
        inertia: bool = False,
        **_ignored,
    ) -> None:
        self.app = QApplication.instance() or QApplication(sys.argv)
        super().__init__()

        self.setWindowTitle("K-Scale MuJoCo Viewer")
        self.resize(width, height)

        self._data = data or mujoco.MjData(model)
        self._ring = ring

        self._viewport = GLViewport(
            model,
            self._data,
            ring=self._ring,
            shadow=shadow,
            reflection=reflection,
            contact_force=contact_force,
            contact_point=contact_point,
            inertia=inertia,
        )
        self.setCentralWidget(self._viewport)

        # status bar for FPS readout
        self.setStatusBar(QStatusBar(self))

        self._scalar_plot = ScalarPlot(history=600, max_curves=24)
        dock = QDockWidget("Scalars", self)
        dock.setWidget(self._scalar_plot)
        self.addDockWidget(Qt.BottomDockWidgetArea, dock)

        if mode == "window":
            self.show()

    @property
    def model(self) -> mujoco.MjModel:
        return self._viewport.model

    @property
    def data(self) -> mujoco.MjData:
        return self._data

    @property
    def cam(self) -> mujoco.MjvCamera:
        return self._viewport.cam

    @property
    def scn(self) -> mujoco.MjvScene:
        return self._viewport.scene

    @property
    def vopt(self) -> mujoco.MjvOption:
        return self._viewport.opt

    def set_mjdata(self, data: mujoco.MjData) -> None:
        self._data = data
        self._viewport.set_mjdata(data)

    def render(self, callback: Callback | None = None) -> None:
        self._viewport.set_callback(callback)
        self._viewport.update()
        self.app.processEvents()

    def read_pixels(self, callback: Callback | None = None) -> np.ndarray:
        self._viewport.set_callback(callback)
        self._viewport.makeCurrent()
        img = self._viewport.grabFramebuffer()
        arr = img.toImage().convertToFormat(4).constBits().asarray(img.height()*img.width()*4)
        return arr.reshape(img.height(), img.width(), 4)[..., :3]        # RGB

    def push_scalars(self, t: float, scalars: dict[str, float]) -> None:
        """Stream one time-stamp worth of scalars to the live plot."""
        self._scalar_plot.update_data(t, scalars)

    def feed(self, frame: Frame) -> None:
        """Push a new physics frame into the ring (cheap, non-blocking)."""
        if self._ring is not None:
            self._ring.push(frame)

    def keyPressEvent(self, ev):                           # type: ignore[override]
        if ev.key() == Qt.Key_R:
            mujoco.mjv_defaultFreeCamera(self.model, self.cam)
            self._viewport.update()
        elif ev.key() in (Qt.Key_Escape, Qt.Key_Q):
            self.close()



class DefaultMujocoViewer:
    """Very small off-screen renderer used for video export."""

    def __init__(self, model: mujoco.MjModel, *, width: int = 640, height: int = 480) -> None:
        self.model = model
        self.data  = mujoco.MjData(model)

        self._w, self._h = width, height
        self._ctx = mujoco.gl_context.GLContext(width, height)
        self._ctx.make_current()

        self.scene = mujoco.MjvScene(model, maxgeom=20_000)
        self.cam   = mujoco.MjvCamera()
        self.opt   = mujoco.MjvOption()
        self.pert  = mujoco.MjvPerturb()
        mujoco.mjv_defaultFreeCamera(model, self.cam)

        self._rect = mujoco.MjrRect(0, 0, width, height)
        self._mjr  = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self._mjr)

    def set_mjdata(self, data: mujoco.MjData) -> None:
        self.data = data

    def render(self, callback: Callback | None = None) -> None:
        mujoco.mjv_updateScene(self.model, self.data,
                               self.opt, self.pert, self.cam,
                               mujoco.mjtCatBit.mjCAT_ALL,
                               self.scene)
        if callback:
            callback(self.model, self.data, self.scene)
        mujoco.mjr_render(self._rect, self.scene, self._mjr)

    def read_pixels(self, callback: Callback | None = None) -> np.ndarray:
        self.render(callback)
        rgb = np.empty((self._h, self._w, 3), dtype=np.uint8)
        mujoco.mjr_readPixels(rgb, None, self._rect, self._mjr)
        return np.flipud(rgb)
