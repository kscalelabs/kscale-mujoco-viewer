# kmv/app/viewer.py
"""
`Viewer` – the *only* class your RL / control loop talks to.

Responsibilities
----------------
* Serialise a compiled `mjModel` to a temp .mjb file.
* Allocate one shared-memory ring per data stream (qpos, qvel, …).
* Create a control pipe and a bounded metrics queue.
* Spawn the GUI process (`worker.entrypoint.run_worker`) with the above handles.
* Provide ergonomic push / poll helpers for the parent process.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence, Literal, Callable
import dataclasses

import mujoco
import numpy as np

import time

import multiprocessing as mp

from kmv.core.types import RenderMode, ViewerConfig
from kmv.core import streams                     # declares default streams
from kmv.ipc.shared_ring import SharedMemoryRing
from kmv.ipc.control import ControlPipe, make_metrics_queue
from kmv.worker.entrypoint import run_worker


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #

def _compile_model_to_mjb(model: mujoco.MjModel) -> Path:
    """Write `model` to a temp .mjb file and return the path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".mjb", delete=False)
    mujoco.mj_saveModel(model, tmp.name, None)
    tmp.close()
    return Path(tmp.name)


def _build_shm_rings(model: mujoco.MjModel) -> dict[str, SharedMemoryRing]:
    """Create rings for every stream defined in `core.schema`."""
    rings: dict[str, SharedMemoryRing] = {}
    for name, shape in streams.default_streams(model).items():
        rings[name] = SharedMemoryRing(create=True, shape=shape)
    return rings


# --------------------------------------------------------------------------- #
#  Public class
# --------------------------------------------------------------------------- #

class QtViewer:
    """
    High-level out-of-process viewer.

    Usage
    -----
    >>> viewer = Viewer(mj_model)
    >>> while training:
    ...     mujoco.mj_step(model, data)
    ...     viewer.push_state(data.qpos, data.qvel, sim_time=data.time)
    ...     forces = viewer.poll_forces()
    """

    # ------------------------------------------------------------------ #

    def __init__(
        self,
        mj_model: mujoco.MjModel,
        *,
        mode: RenderMode = "window",
        # ↓ all public knobs – defaults match ViewerConfig
        width: int  = 900,
        height: int = 550,
        enable_plots: bool = True,

        shadow: bool        = False,
        reflection: bool    = False,
        contact_force: bool = False,
        contact_point: bool = False,
        inertia: bool       = False,

        camera_distance : float | None = None,
        camera_azimuth  : float | None = None,
        camera_elevation: float | None = None,
        camera_lookat   : tuple[float, float, float] | None = None,
        track_body_id   : int | None = None,
    ) -> None:
        if mode not in ("window", "offscreen"):
            raise ValueError(f"unknown render mode {mode!r}")

        # ---- one-line conversion to canonical config --------------------- #
        config = ViewerConfig(
            width           = width,
            height          = height,
            enable_plots    = enable_plots,
            shadow          = shadow,
            reflection      = reflection,
            contact_force   = contact_force,
            contact_point   = contact_point,
            inertia         = inertia,
            camera_distance = camera_distance,
            camera_azimuth  = camera_azimuth,
            camera_elevation= camera_elevation,
            camera_lookat   = camera_lookat,
            track_body_id   = track_body_id,
        )

        self._mode   = mode
        self._config = config        # store for later
        self._tmp_mjb_path = _compile_model_to_mjb(mj_model)

        # ---------- shared memory for bulk state ----------------------- #
        self._rings = _build_shm_rings(mj_model)
        shm_cfg = {
            name: {"name": ring.name, "shape": ring.shape}
            for name, ring in self._rings.items()
        }

        # ---------- control & metrics IPC ------------------------------ #
        self._ctrl      = ControlPipe()
        ctx             = mp.get_context("spawn")
        self._table_q   = make_metrics_queue()
        self._plot_q    = make_metrics_queue()

        # ── NEW: physics-state push counter -------------------------------- #
        self._push_ctr  = 0            # total calls to .push_state()
        self._closed = False           # ← NEW sentinel

        # ---------- spawn GUI process ---------------------------------- #
        self._proc = ctx.Process(
            target=run_worker,
            args=(
                str(self._tmp_mjb_path),
                shm_cfg,
                self._ctrl.sender(),       # control pipe
                self._table_q,             # NEW
                self._plot_q,              # NEW
                config,                    # dataclass – picklable
            ),
            daemon=True,
        )
        self._proc.start()

        # ── wait for the GUI to say "ready" ------------------------------ #
        _t0 = time.perf_counter()
        while True:
            if self._ctrl.poll():               # message waiting
                tag, _ = self._ctrl.recv()
                if tag == "ready":
                    break                       # handshake complete
                if tag == "shutdown":           # early exit in worker
                    raise RuntimeError("Viewer process terminated during start-up")
            if (time.perf_counter() - _t0) > 5.0:
                raise TimeoutError("Viewer did not initialise within 5 s")
            time.sleep(0.01)                    # keep the loop cheap

    # ------------------------------------------------------------------ #
    #  Producer helpers – called from sim loop
    # ------------------------------------------------------------------ #

    @property
    def is_open(self) -> bool:
        """True while the GUI process is alive (or hasn't been closed)."""
        return not self._closed

    def push_state(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        *,
        sim_time: float | int = 0.0,
        xfrc_applied: np.ndarray | None = None,
    ) -> None:
        """Copy MuJoCo state into shared rings (qpos / qvel)."""
        if self._closed:
            return

        # ① keep local running total (cheap & thread-safe under GIL)
        self._push_ctr += 1

        self._rings["qpos"].push(qpos)
        self._rings["qvel"].push(qvel)
        self._rings["sim_time"].push(np.asarray([sim_time], dtype=np.float64))

        if xfrc_applied is not None:
            # If you later add a ring for forces, push here
            pass

        # ② stream the counter to GUI – exactly the same path used for "iters"
        #    (small dict → bounded multiprocessing.Queue)
        self._table_q.put({"Phys Iters": self._push_ctr})

    # ------------------------------------------------------------------ #
    #  New convenience helpers
    # ------------------------------------------------------------------ #

    def push_table_metrics(self, metrics: Mapping[str, float]) -> None:
        """Send key-value pairs to the telemetry table only."""
        if self._closed:
            return
        self._table_q.put(dict(metrics))

    def push_plot_metrics(
        self,
        scalars: Mapping[str, float],
        group: str = "default",
    ) -> None:
        """
        Stream a batch of *scalars* belonging to *group*.

        Parameters
        ----------
        scalars : mapping {name -> value}
        group   : name of the panel to plot into ("physics", "reward", …)
        """
        if self._closed:
            return
        self._plot_q.put({"group": group, "scalars": dict(scalars)})

    # ------------------------------------------------------------------ #
    #  Consumer helper – drag forces coming back
    # ------------------------------------------------------------------ #

    def poll_forces(self) -> np.ndarray | None:
        """
        Non-blocking.  Returns the latest ``xfrc_applied`` array generated by
        mouse interaction in the GUI, or ``None`` if nothing new.
        """
        if self._closed:
            return None

        try:
            out = None
            while self._ctrl.poll():
                tag, payload = self._ctrl.recv()
                if tag == "forces":
                    out = payload
                elif tag == "shutdown":
                    self.close()          # sets _closed
            return out

        except (OSError, EOFError):       # pipe already gone
            self._closed = True
            return None

    # ------------------------------------------------------------------ #
    #  Shutdown
    # ------------------------------------------------------------------ #

    def close(self) -> None:
        """Ask the worker to quit, wait, *then* unlink shared memory."""
        if self._closed:
            return

        if not self._proc.is_alive():
            self._closed = True
            return            # already closed

        # (1) polite SIGTERM (handled inside worker — see patch above)
        self._proc.terminate()
        self._proc.join(timeout=2.0)

        # (2) if still stubborn, kill hard
        if self._proc.is_alive():
            self._proc.kill()
            self._proc.join(timeout=1.0)

        # (3) now we are the *only* process mapped → safe to unlink
        for ring in self._rings.values():
            ring.close()
            ring.unlink()

        self._ctrl.close()
        self._tmp_mjb_path.unlink(missing_ok=True)
        self._closed = True


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