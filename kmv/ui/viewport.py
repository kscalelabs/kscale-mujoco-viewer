# kmv/ui/viewport.py
"""
OpenGL widget that asks MuJoCo to render the current world.

Pure Qt + MuJoCo; no multiprocessing, no shared-memory.
"""

from __future__ import annotations

from typing import Callable

import mujoco
import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtGui  import QSurfaceFormat
from PySide6.QtOpenGLWidgets import QOpenGLWidget


# -- configure default OpenGL format (anti-aliasing, depth, vsync) ------------
_fmt = QSurfaceFormat()
_fmt.setDepthBufferSize(24)
_fmt.setStencilBufferSize(8)
_fmt.setSamples(4)
_fmt.setSwapInterval(1)                # vsync
QSurfaceFormat.setDefaultFormat(_fmt)


class GLViewport(QOpenGLWidget):
    """
    Read-only MuJoCo viewer living entirely in the Qt (GUI) thread.

    Mouse controls
    --------------
    • Drag   : rotate camera
    • Wheel  : zoom
    • Ctrl-drag (L/R) : body perturb (rotate / translate) – generates forces
    """

    # ------------------------------------------------------------------ #

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        shadow: bool = False,
        reflection: bool = False,
        contact_force: bool = False,
        contact_point: bool = False,
        inertia: bool = False,
        parent=None,
    ) -> None:
        super().__init__(parent)

        self.model, self._data = model, data
        self.scene = mujoco.MjvScene(model, maxgeom=20_000)
        self.cam   = mujoco.MjvCamera()
        self.opt   = mujoco.MjvOption()
        self.pert  = mujoco.MjvPerturb()
        mujoco.mjv_defaultFreeCamera(model, self.cam)

        # visual flags
        self.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]     = shadow
        self.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = reflection
        self.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = contact_force
        self.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = contact_point
        self.opt.flags[mujoco.mjtVisFlag.mjVIS_INERTIA]      = inertia

        # callback for overlay rendering
        self._callback: Callable[[mujoco.MjModel, mujoco.MjData, mujoco.MjvScene], None] | None = None

        # mouse state
        self._mouse_btn: int | None = None
        self._last_x = 0.0
        self._last_y = 0.0

    # ------------------------------------------------------------------ #
    #  Public helpers
    # ------------------------------------------------------------------ #

    def set_callback(self, fn: Callable[[mujoco.MjModel, mujoco.MjData, mujoco.MjvScene], None] | None) -> None:
        self._callback = fn

    # ------------------------------------------------------------------ #
    #  OpenGL lifecycle
    # ------------------------------------------------------------------ #

    def initializeGL(self) -> None:
        self._ctx = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)

    def paintGL(self) -> None:
        # MuJoCo expects xfrc_applied to be cleared each frame
        self._data.xfrc_applied[:] = 0
        mujoco.mjv_applyPerturbPose(self.model, self._data, self.pert, 0)
        mujoco.mjv_applyPerturbForce(self.model, self._data, self.pert)

        # ── NEW: stream wrench continuously while dragging ────────────── #
        if self.pert.active and hasattr(self.parent(), "_ctrl_send"):
            # Copy is cheap (nbody × 6 doubles) and keeps the pipe unshared
            self.parent()._ctrl_send.send(("forces",
                                           self._data.xfrc_applied.copy()))

        dpr  = self.devicePixelRatioF()
        rect = mujoco.MjrRect(0, 0,
                              int(self.width() * dpr),
                              int(self.height() * dpr))

        mujoco.mjv_updateScene(
            self.model,
            self._data,
            self.opt,
            self.pert,
            self.cam,
            mujoco.mjtCatBit.mjCAT_ALL,
            self.scene,
        )

        if self._callback:
            self._callback(self.model, self._data, self.scene)

        mujoco.mjr_render(rect, self.scene, self._ctx)

    # ------------------------------------------------------------------ #
    #  Mouse interaction (camera + perturb)
    # ------------------------------------------------------------------ #

    def mousePressEvent(self, ev):                         # type: ignore[override]
        self._mouse_btn = ev.button()
        self._last_x, self._last_y = ev.position().x(), ev.position().y()

        if not (ev.modifiers() & Qt.ControlModifier):
            return

        # Ctrl-click → select MuJoCo body under cursor
        dpr    = self.devicePixelRatioF()
        width  = max(1, int(self.width()  * dpr))
        height = max(1, int(self.height() * dpr))
        aspect = width / height
        relx   = (self._last_x * dpr) / width
        rely   = (height - self._last_y * dpr) / height

        selpnt = np.zeros(3, dtype=np.float64)
        geomid = np.zeros(1, dtype=np.int32)
        flexid = np.zeros(1, dtype=np.int32)
        skinid = np.zeros(1, dtype=np.int32)

        gid = mujoco.mjv_select(
            self.model, self._data, self.opt,
            aspect, relx, rely,
            self.scene, selpnt,
            geomid, flexid, skinid,
        )
        if gid < 0:
            return

        bodyid                = gid
        self.pert.select      = bodyid
        self.pert.skinselect  = int(skinid[0])
        diff                  = selpnt - self._data.xpos[bodyid]
        self.pert.localpos    = self._data.xmat[bodyid].reshape(3, 3) @ diff
        self.pert.active      = (
            mujoco.mjtPertBit.mjPERT_ROTATE
            if self._mouse_btn == Qt.LeftButton else
            mujoco.mjtPertBit.mjPERT_TRANSLATE
        )
        mujoco.mjv_initPerturb(self.model, self._data, self.scene, self.pert)
        self.update()

    def mouseReleaseEvent(self, _ev):                      # type: ignore[override]
        released = self._mouse_btn
        self.pert.active = 0
        self._mouse_btn  = None
        self.update()

        # ── NEW ────────────────────────────────────────────────
        # Flush a single "zero wrench" so the physics loop
        # knows the drag interaction has ended.
        if hasattr(self.parent(), "_ctrl_send"):
            zero_xrfc = np.zeros_like(self._data.xfrc_applied)
            self.parent()._ctrl_send.send(("forces", zero_xrfc))

    def mouseMoveEvent(self, ev):                          # type: ignore[override]
        x, y = ev.position().x(), ev.position().y()
        dx, dy = x - self._last_x, y - self._last_y
        self._last_x, self._last_y = x, y

        if self.pert.active:                               # Ctrl-drag → perturb
            height = max(1, self.height())

            if self.pert.active == mujoco.mjtPertBit.mjPERT_TRANSLATE:
                # ---------------------------------------------------------- #
                # Ctrl-drag           → vertical translate  (MOVE_V)
                # Shift + Ctrl-drag   → horizontal translate (MOVE_H)
                # ---------------------------------------------------------- #
                action = (
                    mujoco.mjtMouse.mjMOUSE_MOVE_H              # ⇧ held  → horizontal
                    if (ev.modifiers() & Qt.ShiftModifier)
                    else mujoco.mjtMouse.mjMOUSE_MOVE_V          # default → vertical
                )
            else:                                               # rotate branch
                action = mujoco.mjtMouse.mjMOUSE_ROTATE_H       # unchanged

            mujoco.mjv_movePerturb(
                self.model, self._data,
                action,
                dx / height, dy / height,
                self.scene, self.pert,
            )
            self.update()
            return

        # normal camera controls
        if self._mouse_btn == Qt.LeftButton:
            self.cam.azimuth   += 0.25 * dx
            self.cam.elevation += 0.25 * dy
            self.cam.elevation = np.clip(self.cam.elevation, -89.9, 89.9)
        elif self._mouse_btn == Qt.RightButton:
            scale = 0.002 * self.cam.distance
            right = np.array([1.0, 0.0, 0.0])
            fwd   = np.array([0.0, 1.0, 0.0])
            self.cam.lookat += (-dx * scale) * right + (dy * scale) * fwd

        self.update()

    # ------------------------------------------------------------------ #
    #  Mouse-wheel zoom  (softer: ~4 % per notch instead of 10 %)
    # ------------------------------------------------------------------ #
    def wheelEvent(self, ev):
        # One standard notch on most mice → angleDelta().y() = ±120
        step = np.sign(ev.angleDelta().y())          # +1 (zoom-in) / −1 (out)
        zoom_factor = 0.99 if step > 0 else 1.01     # ← 4 % change

        # apply & clamp
        self.cam.distance *= zoom_factor
        self.cam.distance = np.clip(self.cam.distance, 0.1, 100.0)
        self.update()
