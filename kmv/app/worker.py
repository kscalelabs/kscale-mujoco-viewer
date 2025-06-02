"""
Worker process.  Receives an XML file-path, opens its own MjModel/MjData,
pulls frames from shared memory and renders at 60 Hz.
"""

import sys, multiprocessing as mp, pathlib
import mujoco, numpy as np
from PySide6.QtWidgets import QApplication
from PySide6.QtCore    import QTimer
from multiprocessing.connection import Connection

from kmv.app.viewer            import QtViewer
from kmv.ipc.shared_ringbuffer import SharedRing


def _worker(model_path: str,
            shm_name: str,
            nq: int,
            nv: int,
            ctrl_send: Connection,               # ✅ receive the Connection
            metrics_queue: mp.Queue = None) -> None:
    # Detect file format and load model accordingly
    model_path = pathlib.Path(model_path)
    if model_path.suffix.lower() == '.mjb':
        model = mujoco.MjModel.from_binary_path(str(model_path))
    else:
        # Default to XML loading for .xml files or unknown extensions
        model = mujoco.MjModel.from_xml_path(str(model_path))
    
    data  = mujoco.MjData(model)

    ring  = SharedRing(name=shm_name, create=False, nq=nq, nv=nv)

    app    = QApplication(sys.argv)
    viewer = QtViewer(model, data, enable_plots=False)      # normal widget
    viewer._ctrl_send = ctrl_send          # ← enables force round-trip

    # repaint timer ─────────────────────────────────────────────────────────
    t = QTimer()
    t.setInterval(16)                # ≈60 Hz
    t.timeout.connect(viewer.update)
    t.start()

    # telemetry timer ───────────────────────────────────────────────────────
    if metrics_queue:
        def check_metrics():
            try:
                while not metrics_queue.empty():
                    metrics = metrics_queue.get_nowait()
                    viewer.update_telemetry(metrics)
            except:
                pass
        
        metrics_timer = QTimer()
        metrics_timer.setInterval(50)  # 20 Hz for telemetry updates
        metrics_timer.timeout.connect(check_metrics)
        metrics_timer.start()

    # copy latest qpos/qvel before every paint
    def sync():
        qpos, qvel, sim_time = ring.latest()
        data.qpos[:] = qpos
        data.qvel[:] = qvel
        data.time    = sim_time          # <-- critical
        mujoco.mj_forward(model, data)

    viewer.before_paint_callback = sync

    exit_code = app.exec()
    ctrl_send.send(("shutdown", exit_code))
