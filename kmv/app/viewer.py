"""Qt-based Mujoco viewer that runs in a separate process.

Exposes two viewer handles:
• QtViewer: Push physics state & fetch drag forces from the parent.
• DefaultMujocoViewer: Simple headless off-screen renderer for video.
"""

import multiprocessing as mp
import tempfile
import time
from pathlib import Path
from typing import Callable, Mapping

import mujoco
import numpy as np

from kmv.core import streams
from kmv.core.types import RenderMode, ViewerConfig
from kmv.ipc.control import ControlPipe, make_metrics_queue
from kmv.ipc.shared_ring import SharedMemoryRing
from kmv.worker.entrypoint import run_worker


def _compile_model_to_mjb(model: mujoco.MjModel) -> Path:
    """Write `model` to a temp .mjb file and return the path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".mjb", delete=False)
    mujoco.mj_saveModel(model, tmp.name, None)
    tmp.close()
    return Path(tmp.name)


def _build_shm_rings(model: mujoco.MjModel) -> dict[str, SharedMemoryRing]:
    """Create rings for every stream defined in `core.streams`."""
    rings: dict[str, SharedMemoryRing] = {}
    for name, shape in streams.default_streams(model).items():
        rings[name] = SharedMemoryRing(create=True, shape=shape)
    return rings


class QtViewer:
    """Viewer class for the Qt application.

    Creates a new process in which to run the GUI
    in order to avoid blocking the main thread.
    """

    def __init__(
        self,
        mj_model: mujoco.MjModel,
        *,
        mode: RenderMode = "window",
        # ↓ all public knobs – defaults match ViewerConfig
        width: int = 900,
        height: int = 550,
        enable_plots: bool = True,
        shadow: bool = False,
        reflection: bool = False,
        contact_force: bool = False,
        contact_point: bool = False,
        inertia: bool = False,
        camera_distance: float | None = None,
        camera_azimuth: float | None = None,
        camera_elevation: float | None = None,
        camera_lookat: tuple[float, float, float] | None = None,
        track_body_id: int | None = None,
    ) -> None:
        if mode not in ("window", "offscreen"):
            raise ValueError(f"unknown render mode {mode!r}")

        config = ViewerConfig(
            width=width,
            height=height,
            enable_plots=enable_plots,
            shadow=shadow,
            reflection=reflection,
            contact_force=contact_force,
            contact_point=contact_point,
            inertia=inertia,
            camera_distance=camera_distance,
            camera_azimuth=camera_azimuth,
            camera_elevation=camera_elevation,
            camera_lookat=camera_lookat,
            track_body_id=track_body_id,
        )

        self._mode = mode
        self._config = config
        self._tmp_mjb_path = _compile_model_to_mjb(mj_model)
        self._rings = _build_shm_rings(mj_model)
        shm_cfg = {name: {"name": ring.name, "shape": ring.shape} for name, ring in self._rings.items()}
        self._ctrl = ControlPipe()
        ctx = mp.get_context()
        self._table_q = make_metrics_queue()
        self._plot_q = make_metrics_queue()
        self._push_ctr = 0
        self._closed = False

        # Start the new GUI process
        self._proc = ctx.Process(
            target=run_worker,
            args=(
                str(self._tmp_mjb_path),
                shm_cfg,
                self._ctrl.sender(),
                self._table_q,
                self._plot_q,
                config,
            ),
            daemon=True,
        )
        self._proc.start()

        # Wait for the viewer to be ready before continuing
        _t0 = time.perf_counter()
        while True:
            if self._ctrl.poll():
                tag, _ = self._ctrl.recv()
                match tag:
                    case "ready":
                        break
                    case "shutdown":
                        raise RuntimeError("Viewer process terminated during start-up")
            if (time.perf_counter() - _t0) > 5.0:
                raise TimeoutError("Viewer did not initialise within 5 s")
            time.sleep(0.01)

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
    ) -> None:
        """Copy MuJoCo state into shared rings (qpos / qvel)."""
        if self._closed:
            return

        self._push_ctr += 1
        self._rings["qpos"].push(qpos)
        self._rings["qvel"].push(qvel)
        self._rings["sim_time"].push(np.asarray([sim_time], dtype=np.float64))
        self._table_q.put({"Phys Iters": self._push_ctr})

    def push_table_metrics(self, metrics: Mapping[str, float]) -> None:
        """Send key-value pairs to the telemetry table."""
        if self._closed:
            return
        self._table_q.put(dict(metrics))

    def push_plot_metrics(
        self,
        scalars: Mapping[str, float],
        group: str = "default",
    ) -> None:
        """Stream a batch of scalars belonging to a plot group."""
        if self._closed:
            return
        self._plot_q.put({"group": group, "scalars": dict(scalars)})

    def drain_control_pipe(self) -> np.ndarray | None:
        """Return the latest push forces array.

        Generated by mouse interaction in the GUI, or ``None`` if nothing new.
        """
        if self._closed:
            return None

        try:
            out = None
            while self._ctrl.poll():
                tag, payload = self._ctrl.recv()
                match tag:
                    case "forces":
                        out = payload
                    case "shutdown":
                        self.close()
            return out

        except (OSError, EOFError):
            self._closed = True
            return None

    def close(self) -> None:
        """Ask the worker to quit, wait, *then* unlink shared memory.

        It is important to do this cleanup to prevent the viewer from leaking
        shared memory.
        """
        if self._closed:
            return

        if not self._proc.is_alive():
            self._closed = True
            return

        self._proc.terminate()
        self._proc.join(timeout=2.0)

        if self._proc.is_alive():
            self._proc.kill()
            self._proc.join(timeout=1.0)

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
