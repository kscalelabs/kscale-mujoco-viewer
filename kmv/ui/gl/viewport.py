"""Tiny QOpenGLWidget that asks MuJoCo to draw the current world."""

from __future__ import annotations
import time
from typing import Callable

import mujoco
import numpy as np
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtOpenGLWidgets import QOpenGLWidget

from kmv.core.buffer import RingBuffer
from kmv.core.types import Frame


_fmt = QSurfaceFormat()
_fmt.setDepthBufferSize(24)
_fmt.setStencilBufferSize(8)
_fmt.setSamples(4)
_fmt.setSwapInterval(1)            # v-sync
QSurfaceFormat.setDefaultFormat(_fmt)


class GLViewport(QOpenGLWidget):
    """
    Read-only MuJoCo viewer living entirely in the Qt (GUI) thread.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        ringbuffer: RingBuffer[Frame] | None = None,
        shadow: bool = False,
        reflection: bool = False,
        contact_force: bool = False,
        contact_point: bool = False,
        inertia: bool = False,
        parent=None,
    ) -> None:
        super().__init__(parent)

        self.model, self._data_ptr = model, data
        self.scene = mujoco.MjvScene(model, maxgeom=20_000)
        self.cam   = mujoco.MjvCamera()
        self.opt   = mujoco.MjvOption()
        self.pert  = mujoco.MjvPerturb()
        mujoco.mjv_defaultFreeCamera(model, self.cam)

        self._set_vis_flags(
            shadow=shadow,
            reflection=reflection,
            contact_force=contact_force,
            contact_point=contact_point,
            inertia=inertia,
        )

        self._ringbuffer = ringbuffer
        self._callback: Callable | None = None
        self._fps_timer = time.time()
        self._frame_ctr = 0

        self._mouse_btn: int | None = None
        self._last_x  = 0.0
        self._last_y  = 0.0
        self._sel_body: int = -1

        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self.update)
        self._timer.start()

    def set_callback(
        self,
        fn: Callable[[mujoco.MjModel, mujoco.MjData, mujoco.MjvScene], None] | None,
    ) -> None:
        self._callback = fn

    def set_mjdata(self, data: mujoco.MjData) -> None:
        self._data_ptr = data

    def _set_vis_flags(
        self,
        *,
        shadow: bool,
        reflection: bool,
        contact_force: bool,
        contact_point: bool,
        inertia: bool,
    ) -> None:
        """
        Enable/disable common visual features but stay compatible with
        different MuJoCo versions:

        * contact_force / contact_point / inertia  →  mjtVisFlag  (option.flags)
        * shadow / reflection                      →  mjtRndFlag  (scene.flags)  in v3.x+
                                                    →  mjtVisFlag              in older builds
        """

        def _set_flag(container, enum_cls, name: str, enabled: bool) -> None:
            idx = getattr(enum_cls, name, None)
            if idx is not None:
                container[idx] = int(enabled)

        vis = mujoco.mjtVisFlag
        _set_flag(self.opt.flags, vis, "mjVIS_CONTACTFORCE", contact_force)
        _set_flag(self.opt.flags, vis, "mjVIS_CONTACTPOINT", contact_point)
        _set_flag(self.opt.flags, vis, "mjVIS_INERTIA",       inertia)

        rnd = mujoco.mjtRndFlag
        if hasattr(rnd, "mjRND_SHADOW"):
            _set_flag(self.scene.flags, rnd, "mjRND_SHADOW",      shadow)
            _set_flag(self.scene.flags, rnd, "mjRND_REFLECTION",  reflection)
        else:                                  # fallback for older MuJoCo
            _set_flag(self.opt.flags, vis, "mjVIS_SHADOW",     shadow)
            _set_flag(self.opt.flags, vis, "mjVIS_REFLECTION", reflection)

    def _pick_body(self, x: float, y: float) -> int:
        """
        Return the id of the body under the cursor (or -1).

        MuJoCo ≥ 3.1 signature:
            mjv_select(model, data, vopt,
                       aspect, relx, rely,
                       scene, selpnt,
                       geomid, flexid, skinid)
        """

        dpr    = self.devicePixelRatioF()
        width  = max(1, int(self.width()  * dpr))
        height = max(1, int(self.height() * dpr))
        aspect = width / height

        relx = (x * dpr) / width
        rely = (height - y * dpr) / height

        selpnt = np.zeros(3, dtype=np.float64)
        geomid = np.zeros(1, dtype=np.int32)
        flexid = np.zeros(1, dtype=np.int32)
        skinid = np.zeros(1, dtype=np.int32)

        gid = mujoco.mjv_select(
            self.model,
            self._data_ptr,
            self.opt,
            aspect,
            relx,
            rely,
            self.scene,
            selpnt,
            geomid,
            flexid,
            skinid,
        )

        return -1 if gid < 0 else int(self.model.geom_bodyid[gid])


    def initializeGL(self) -> None:
        self.ctx = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)

    def paintGL(self) -> None:
        if self._ringbuffer is not None:
            frame = self._ringbuffer.latest()
            if frame is not None:
                self._data_ptr.qpos[:] = frame.qpos
                self._data_ptr.qvel[:] = frame.qvel
                if frame.xfrc_applied is not None:
                    self._data_ptr.xfrc_applied[:] = frame.xfrc_applied
                mujoco.mj_forward(self.model, self._data_ptr)

        self._data_ptr.xfrc_applied[:] = 0          # clear previous frame
        mujoco.mjv_applyPerturbPose(self.model, self._data_ptr, self.pert, 0)
        mujoco.mjv_applyPerturbForce(self.model, self._data_ptr, self.pert)

        dpr  = self.devicePixelRatioF()
        rect = mujoco.MjrRect(0, 0,
                              int(self.width()*dpr),
                              int(self.height()*dpr))

        mujoco.mjv_updateScene(
            self.model,
            self._data_ptr,
            self.opt,
            self.pert,
            self.cam,
            mujoco.mjtCatBit.mjCAT_ALL,
            self.scene,
        )

        if self._callback:
            self._callback(self.model, self._data_ptr, self.scene)

        mujoco.mjr_render(rect, self.scene, self.ctx)

        self._frame_ctr += 1
        if time.time() - self._fps_timer >= 1.0:
            if self._ringbuffer is not None:
                pushes = self._ringbuffer.push_count
                pops   = self._ringbuffer.pop_count
                dropped = pushes - pops
                backlog = len(self._ringbuffer)
                msg = (
                    f"{self._frame_ctr} FPS   "
                    f"P:{pushes}  C:{pops}  Δ:{dropped}  len:{backlog}"
                )
            else:
                msg = f"{self._frame_ctr} FPS"
            self.window().statusBar().showMessage(msg)

            self._frame_ctr = 0
            self._fps_timer = time.time()

    def mousePressEvent(self, ev):                         # type: ignore[override]
        """Start camera drag or (Ctrl + button) perturb drag."""
        self._mouse_btn = ev.button()
        self._last_x, self._last_y = ev.position().x(), ev.position().y()

        if not (ev.modifiers() & Qt.ControlModifier):
            return

        dpr     = self.devicePixelRatioF()
        width   = max(1, int(self.width()  * dpr))
        height  = max(1, int(self.height() * dpr))
        aspect  = width / height
        relx    = (self._last_x * dpr) / width
        rely    = (height - self._last_y * dpr) / height

        selpnt  = np.zeros(3, dtype=np.float64)
        geomid  = np.zeros(1, dtype=np.int32)
        flexid  = np.zeros(1, dtype=np.int32)
        skinid  = np.zeros(1, dtype=np.int32)

        gid = mujoco.mjv_select(
            self.model, self._data_ptr, self.opt,
            aspect, relx, rely,
            self.scene, selpnt,
            geomid, flexid, skinid,
        )

        if gid < 0:
            return

        bodyid = gid
        self.pert.select = bodyid
        self.pert.skinselect = int(skinid[0])

        diff = selpnt - self._data_ptr.xpos[bodyid]
        self.pert.localpos = self._data_ptr.xmat[bodyid].reshape(3, 3) @ diff

        # choose perturb mode from the mouse button
        if   self._mouse_btn == Qt.LeftButton:
            self.pert.active = mujoco.mjtPertBit.mjPERT_ROTATE
        elif self._mouse_btn == Qt.RightButton:
            self.pert.active = mujoco.mjtPertBit.mjPERT_TRANSLATE

        mujoco.mjv_initPerturb(self.model, self._data_ptr, self.scene, self.pert)
        self.update()

    def mouseReleaseEvent(self, _ev):                       # type: ignore[override]
        """Stop drag (camera or perturb)."""
        self.pert.active = 0
        self._mouse_btn  = None
        self.update()

    def mouseMoveEvent(self, ev):                          # type: ignore[override]
        x, y = ev.position().x(), ev.position().y()
        dx, dy = x - self._last_x, y - self._last_y
        self._last_x, self._last_y = x, y

        if self.pert.active:
            action = (mujoco.mjtMouse.mjMOUSE_ROTATE_H
                      if self.pert.active == mujoco.mjtPertBit.mjPERT_ROTATE
                      else mujoco.mjtMouse.mjMOUSE_MOVE_H)

            # scale to NDC the way MuJoCo expects
            height = max(1, self.height())
            mujoco.mjv_movePerturb(
                self.model, self._data_ptr,
                action,
                dx / height,
                dy / height,
                self.scene,
                self.pert,
            )
            self.update()
            return

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

    def wheelEvent(self, ev):
        self.cam.distance *= 0.9 if ev.angleDelta().y() > 0 else 1.1
        self.cam.distance = np.clip(self.cam.distance, 0.1, 100.0)
        self.update()
