"""
Main-process handle.  Spawns the worker process and streams frames through
shared memory.
"""

import os, signal, multiprocessing as mp, pathlib
import mujoco
import numpy as np

from kmv.ipc.shared_ringbuffer import SharedRing
from kmv.app.worker            import _worker


class RemoteViewer:
    def __init__(self, mj_model: mujoco.MjModel, model_path: str | pathlib.Path):
        """
        Initialize RemoteViewer.
        
        Args:
            mj_model: The MuJoCo model instance
            model_path: Path to the model file (supports both .xml and .mjb formats)
        """
        model_path = str(model_path)

        # ── shared ring ───────────────────────────────────────────────
        self._ring = SharedRing(create=True, nq=mj_model.nq, nv=mj_model.nv)

        # ── telemetry queue ───────────────────────────────────────────
        ctx = mp.get_context("spawn")
        self._metrics_queue = ctx.Queue()

        # ── one-way pipe (worker → sim) ───────────────────────────────
        parent_conn, child_conn = ctx.Pipe(duplex=False)   # Connection objects

        # ── launch worker ─────────────────────────────────────────────
        self._proc = ctx.Process(
            target=_worker,
            args=(model_path,
                  self._ring.name,
                  mj_model.nq,
                  mj_model.nv,
                  child_conn,                    # ✅ pass the Connection
                  self._metrics_queue),
            daemon=True,
        )
        self._proc.start()
        child_conn.close()                       # keep one end per process
        self._ctrl_recv = parent_conn            # <-- fixes AttributeError

    # ── producer API ───────────────────────────────────────────────────────
    def push_frame(self, *, qpos, qvel, sim_time: float | int = 0.0):
        self._ring.push(qpos, qvel, sim_time)
        
    def push_metrics(self, metrics: dict[str, float]) -> None:
        """Update the live telemetry table."""
        self._metrics_queue.put(metrics)

    def poll_forces(self) -> np.ndarray | None:
        """Return latest xfrc_applied from GUI, or None (non-blocking)."""
        out = None
        try:
            while self._ctrl_recv.poll():
                tag, payload = self._ctrl_recv.recv()
                if tag == "forces":
                    out = payload           # keep newest only
                elif tag == "shutdown":
                    self.close()
        except EOFError:                     # worker crashed
            self.close()
        return out

    def close(self):
        if self._proc.is_alive():
            os.kill(self._proc.pid, signal.SIGTERM)
            self._proc.join()
        self._ring.close()
