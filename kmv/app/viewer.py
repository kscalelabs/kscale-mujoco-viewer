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
from kmv.ui.chrome.statusbar import SimulationStatusBar
from kmv.core.types import RenderMode, Frame
from kmv.core.buffer import RingBuffer


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
        self._ringbuffer: RingBuffer[Frame] = RingBuffer(size=8)        # fixed, viewer-private

        self._viewport = GLViewport(
            model,
            self._data,
            ringbuffer=self._ringbuffer,
            shadow=shadow,
            reflection=reflection,
            contact_force=contact_force,
            contact_point=contact_point,
            inertia=inertia,
        )
        self.setCentralWidget(self._viewport)

        # status bar for FPS readout and timing information
        status_bar = QStatusBar(self)
        self.setStatusBar(status_bar)
        self._status_bar_manager = SimulationStatusBar(status_bar)

        self._scalar_plot = ScalarPlot(history=600, max_curves=24)
        dock = QDockWidget("Scalars", self)
        dock.setWidget(self._scalar_plot)
        self.addDockWidget(Qt.BottomDockWidgetArea, dock)

        # Time tracking for absolute time calculation
        self._time_offset: float = 0.0
        self._last_sim_time: float = 0.0
        self.absolute_sim_time: float = 0.0  # Public attribute as requested

        # Connect viewport to status bar manager
        self._viewport.set_status_bar_manager(self._status_bar_manager)

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

    def _calculate_absolute_time(self, sim_time: float) -> float:
        """Calculate absolute time, handling resets by maintaining an offset."""
        if sim_time < self._last_sim_time - 1e-9:  # detect reset
            self._time_offset += self._last_sim_time
        self._last_sim_time = sim_time
        self.absolute_sim_time = self._time_offset + sim_time
        return self.absolute_sim_time

    def push_mujoco_frame(
        self, 
        *, 
        qpos: np.ndarray, 
        qvel: np.ndarray, 
        sim_time: float,
        xfrc_applied: np.ndarray | None = None
    ) -> None:
        """Append one physics frame (qpos/qvel) to the internal queue."""
        frame = Frame(qpos=qpos, qvel=qvel, xfrc_applied=xfrc_applied)
        self._ringbuffer.push(frame)
        
        # Update timing information
        absolute_time = self._calculate_absolute_time(sim_time)
        self._status_bar_manager.set_sim_time(sim_time)
        self._status_bar_manager.set_absolute_sim_time(absolute_time)

    def push_scalar(self, sim_time: float, scalars: dict[str, float]) -> None:
        """Stream scalar values for live plotting."""
        absolute_time = self._calculate_absolute_time(sim_time)
        self._scalar_plot.update_data(absolute_time, scalars)

    def update(self, callback: Callback | None = None) -> np.ndarray:
        """
        Redraw the scene, pump the Qt event loop, and return the current
        `xfrc_applied` array so the RL loop can copy it back into the sim.
        """
        self._viewport.set_callback(callback)
        self._viewport.update()
        self.app.processEvents()
        return self.data.xfrc_applied.copy()

    # ----------------------------------------------------------------------

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
