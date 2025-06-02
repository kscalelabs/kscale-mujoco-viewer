"""
Main-process handle.  Spawns the worker process and streams frames through
shared memory.
"""

import os, signal, multiprocessing as mp, pathlib
import mujoco

from kmv.ipc.shared_ringbuffer import SharedRing
from kmv.app.worker            import _worker


class RemoteViewer:
    def __init__(self, mj_model: mujoco.MjModel, xml_path: str | pathlib.Path):
        xml_path = str(xml_path)

        # shared ring
        self._ring = SharedRing(create=True, nq=mj_model.nq, nv=mj_model.nv)

        # metrics queue for telemetry
        ctx = mp.get_context("spawn")
        self._metrics_queue = ctx.Queue()

        # one-way control pipe (from worker → sim)
        parent_conn, child_conn = ctx.Pipe(duplex=False)

        # launch worker
        self._proc = ctx.Process(
            target=_worker,
            args=(xml_path,
                  self._ring.name,
                  mj_model.nq,
                  mj_model.nv,
                  child_conn.fileno(),
                  self._metrics_queue),
            daemon=True,
        )
        self._proc.start()
        child_conn.close()
        self._ctrl_recv = parent_conn

    # ── producer API ───────────────────────────────────────────────────────
    def push_frame(self, *, qpos, qvel, sim_time: float | int = 0.0):
        self._ring.push(qpos, qvel, sim_time)
        
    def push_metrics(self, metrics: dict[str, float]) -> None:
        """Update the live telemetry table."""
        self._metrics_queue.put(metrics)

    def poll_forces(self):                      # forces not implemented yet
        while self._ctrl_recv.poll():
            tag, _ = self._ctrl_recv.recv()
            if tag == "shutdown":
                self.close()
        return None

    def close(self):
        if self._proc.is_alive():
            os.kill(self._proc.pid, signal.SIGTERM)
            self._proc.join()
        self._ring.close()
