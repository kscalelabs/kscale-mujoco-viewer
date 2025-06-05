"""OpenGL viewport widget for MuJoCo scenes.

Lives entirely in the GUI thread: renders the current `MjData`, handles mouse
interaction, and streams drag forces back to the parent process.
"""

from typing import Callable, Optional

import mujoco
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QMouseEvent, QSurfaceFormat, QWheelEvent
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QWidget

_fmt = QSurfaceFormat()
_fmt.setDepthBufferSize(24)
_fmt.setStencilBufferSize(8)
_fmt.setSamples(4)
_fmt.setSwapInterval(1)
QSurfaceFormat.setDefaultFormat(_fmt)


class GLViewport(QOpenGLWidget):
    """Read-only MuJoCo viewport running inside the Qt event loop."""

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
        on_forces: Optional[Callable[[np.ndarray], None]] = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        # MuJoCo scene
        self.model, self._data = model, data
        self.scene = mujoco.MjvScene(model, maxgeom=20_000)
        self.cam = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()
        self.pert = mujoco.MjvPerturb()
        mujoco.mjv_defaultFreeCamera(model, self.cam)

        # visual flags
        self.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = shadow
        self.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = reflection
        self.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = contact_force
        self.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = contact_point
        self.opt.flags[mujoco.mjtVisFlag.mjVIS_INERTIA] = inertia

        # callback for overlay rendering
        self._callback: Callable[[mujoco.MjModel, mujoco.MjData, mujoco.MjvScene], None] | None = None

        # forces callback
        self._on_forces = on_forces

        # mouse state
        from PySide6.QtCore import Qt as _QtAlias  # local alias to avoid long lines

        self._mouse_btn: _QtAlias.MouseButton | None = None
        self._last_x = 0.0
        self._last_y = 0.0

    def set_callback(self, fn: Callable[[mujoco.MjModel, mujoco.MjData, mujoco.MjvScene], None] | None) -> None:
        """Register a per-frame overlay callback or `None` to clear it.

        TODO: Figure out how to do this for inter-process communication.
        """
        self._callback = fn

    def initializeGL(self) -> None:  # noqa: N802
        """Create the MuJoCo rendering context."""
        self._ctx = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)

    def paintGL(self) -> None:  # noqa: N802
        """Render a frame and emit drag forces if any."""
        # MuJoCo expects xfrc_applied to be cleared each frame
        self._data.xfrc_applied[:] = 0
        mujoco.mjv_applyPerturbPose(self.model, self._data, self.pert, 0)
        mujoco.mjv_applyPerturbForce(self.model, self._data, self.pert)

        if self.pert.active and self._on_forces is not None:
            self._on_forces(self._data.xfrc_applied.copy())

        dpr = self.devicePixelRatioF()
        rect = mujoco.MjrRect(0, 0, int(self.width() * dpr), int(self.height() * dpr))

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

    def mousePressEvent(self, ev: QMouseEvent) -> None:  # noqa: N802
        """Start drag or body-perturb interaction (Ctrl-click)."""
        self._mouse_btn = ev.button()
        self._last_x, self._last_y = ev.position().x(), ev.position().y()

        if not (ev.modifiers() & Qt.KeyboardModifier.ControlModifier):
            return

        # Ctrl-click: select MuJoCo body under cursor
        dpr = self.devicePixelRatioF()
        width = max(1, int(self.width() * dpr))
        height = max(1, int(self.height() * dpr))
        aspect = width / height
        relx = (self._last_x * dpr) / width
        rely = (height - self._last_y * dpr) / height

        selpnt = np.zeros(3, dtype=np.float64)
        geomid = np.zeros(1, dtype=np.int32)
        flexid = np.zeros(1, dtype=np.int32)
        skinid = np.zeros(1, dtype=np.int32)

        gid = mujoco.mjv_select(
            self.model,
            self._data,
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
        if gid < 0:
            return

        bodyid = gid
        self.pert.select = bodyid
        self.pert.skinselect = int(skinid[0])
        diff = selpnt - self._data.xpos[bodyid]
        self.pert.localpos = self._data.xmat[bodyid].reshape(3, 3) @ diff
        self.pert.active = (
            mujoco.mjtPertBit.mjPERT_ROTATE
            if self._mouse_btn == Qt.MouseButton.LeftButton
            else mujoco.mjtPertBit.mjPERT_TRANSLATE
        )
        mujoco.mjv_initPerturb(self.model, self._data, self.scene, self.pert)
        self.update()

    def mouseReleaseEvent(self, _ev: QMouseEvent) -> None:  # noqa: N802
        """End drag / perturb and send a zero-force flush."""
        self.pert.active = 0
        self._mouse_btn = None
        self.update()

        # Upon mouse release, flush a single "zero wrench" so the physics loop
        # knows the drag interaction has ended.
        if self._on_forces is not None:
            zero_xrfc = np.zeros_like(self._data.xfrc_applied)
            self._on_forces(zero_xrfc)

    def mouseMoveEvent(self, ev: QMouseEvent) -> None:  # noqa: N802
        """Handle camera orbit, pan, and active perturb motion."""
        x, y = ev.position().x(), ev.position().y()
        dx, dy = x - self._last_x, y - self._last_y
        self._last_x, self._last_y = x, y

        if self.pert.active:
            height = max(1, self.height())

            if self.pert.active == mujoco.mjtPertBit.mjPERT_TRANSLATE:
                # Ctrl pressed: move vertically
                # Shift + Ctrl pressed: move horizontally
                action = (
                    mujoco.mjtMouse.mjMOUSE_MOVE_H
                    if (ev.modifiers() & Qt.KeyboardModifier.ShiftModifier)
                    else mujoco.mjtMouse.mjMOUSE_MOVE_V
                )
            else:
                action = mujoco.mjtMouse.mjMOUSE_ROTATE_H

            mujoco.mjv_movePerturb(
                self.model,
                self._data,
                action,
                dx / height,
                dy / height,
                self.scene,
                self.pert,
            )
            self.update()
            return

        # Camera controls
        if self._mouse_btn == Qt.MouseButton.LeftButton:
            self.cam.azimuth -= 0.25 * dx
            self.cam.elevation -= 0.25 * dy
            self.cam.elevation = np.clip(self.cam.elevation, -89.9, 89.9)
        elif self._mouse_btn == Qt.MouseButton.RightButton:
            scale = 0.002 * self.cam.distance
            right = np.array([1.0, 0.0, 0.0])
            fwd = np.array([0.0, 1.0, 0.0])
            self.cam.lookat += (-dx * scale) * right + (dy * scale) * fwd

        self.update()

    def wheelEvent(self, ev: QWheelEvent) -> None:  # noqa: N802
        """Zoom the free camera in/out."""
        step = np.sign(ev.angleDelta().y())
        zoom_factor = 0.99 if step > 0 else 1.01

        self.cam.distance *= zoom_factor
        self.cam.distance = np.clip(self.cam.distance, 0.1, 100.0)
        self.update()
