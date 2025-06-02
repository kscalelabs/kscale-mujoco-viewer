# kmv/worker/entrypoint.py
"""
Entrypoint executed **inside** the GUI process.

It is launched by `kmv.app.viewer.Viewer` via `multiprocessing.Process`.
"""

from __future__ import annotations

import sys
import pathlib
from typing import Any

import mujoco
import numpy as np

from multiprocessing.connection import Connection
from multiprocessing import Queue

from PySide6.QtWidgets import QApplication
from PySide6.QtCore    import QTimer

from kmv.ipc.state       import SharedArrayRing
from kmv.worker.window   import ViewerWindow      # will be implemented next


# --------------------------------------------------------------------------- #
#  run_worker – public entrypoint
# --------------------------------------------------------------------------- #

def run_worker(
    model_path: str,
    shm_cfg: dict[str, dict],    # {"qpos": {"name": str, "shape": tuple}, …}
    ctrl_send: Connection,       # write-only end (forces → parent, shutdown)
    table_q:  Queue,             # NEW
    plot_q:   Queue,             # NEW
    view_opts: dict[str, Any],   # forwarded viewer flags
) -> None:
    """
    Spawned via `multiprocessing.Process( target=run_worker, ... )`.
    """
    # ---- 1.  Load MuJoCo model --------------------------------------- #
    model_path = pathlib.Path(model_path)
    if model_path.suffix.lower() == ".mjb":
        model = mujoco.MjModel.from_binary_path(str(model_path))
    else:
        model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    # ---- 2.  Attach to shared rings ---------------------------------- #
    rings = {
        name: SharedArrayRing(create=False, **cfg)
        for name, cfg in shm_cfg.items()
    }

    # ---- 3.  Qt application & window -------------------------------- #
    app    = QApplication.instance() or QApplication(sys.argv)
    window = ViewerWindow(model, data, rings,
                          table_q=table_q, plot_q=plot_q,
                          view_opts=view_opts)
    window._ctrl_send = ctrl_send      # expose pipe to viewport for forces

    # ── let the parent know the GUI is ready --------------------------- #
    try:
        ctrl_send.send(("ready", None))
    except (BrokenPipeError, EOFError):
        # parent already quit – just keep going so Qt can shut down cleanly
        pass

    # ---- 4.  Graphics timer (≈60 Hz) -------------------------------- #
    gfx_timer = QTimer()
    gfx_timer.setInterval(16)
    gfx_timer.timeout.connect(window.step_and_draw)
    gfx_timer.start()

    # ---- 5.  Event-loop --------------------------------------------- #
    exit_code = app.exec()

    # ---- 6.  Notify parent we're done ------------------------------- #
    try:
        ctrl_send.send(("shutdown", exit_code))
    except (BrokenPipeError, EOFError):
        pass   # parent already quit
