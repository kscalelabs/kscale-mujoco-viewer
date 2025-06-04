"""GUI-worker entry point.

Executed in a forked process by `QtViewer`; loads the model, maps the
shared-memory rings, spins up the `ViewerWindow`, and runs the Qt event loop.
"""

import pathlib
import signal
import sys
from multiprocessing import Queue
from multiprocessing.connection import Connection

import mujoco
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from kmv.core.types import ViewerConfig
from kmv.ipc.shared_ring import SharedMemoryRing
from kmv.worker.window import ViewerWindow

TARGET_FPS = 60
GUI_TIMER_INTERVAL_MS = round(1000 / TARGET_FPS)


def run_worker(
    model_path: str,
    shm_cfg: dict[str, dict],
    ctrl_send: Connection,
    table_q: Queue,
    plot_q: Queue,
    view_conf: ViewerConfig,
) -> None:
    """Boot the GUI process and run its Qt event-loop."""
    # Load the model
    model_path = pathlib.Path(model_path)
    if model_path.suffix.lower() == ".mjb":
        model = mujoco.MjModel.from_binary_path(str(model_path))
    else:
        model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    # Set up shared memory
    rings = {name: SharedMemoryRing(create=False, **cfg) for name, cfg in shm_cfg.items()}

    # Start Qt
    app = QApplication.instance() or QApplication(sys.argv)
    window = ViewerWindow(model, data, rings, table_q=table_q, plot_q=plot_q, ctrl_send=ctrl_send, view_conf=view_conf)

    # Notify the parent that the worker is ready
    # The parent waits on this message before continuing
    # Allows the viewer to start up before the physics loop starts
    try:
        ctrl_send.send(("ready", None))
    except (BrokenPipeError, EOFError):
        # Parent already quit – just keep going so Qt can shut down cleanly
        pass

    def _sigterm_handler(_signum: int, _frame: object) -> None:
        app.quit()

    signal.signal(signal.SIGTERM, _sigterm_handler)

    # Set up the graphics timer
    gfx_timer = QTimer()
    gfx_timer.setInterval(GUI_TIMER_INTERVAL_MS)
    gfx_timer.timeout.connect(window.step_and_draw)
    gfx_timer.start()

    # Run the event loop
    exit_code = 0
    try:
        exit_code = app.exec()
    finally:
        # Detach from shared memory first
        for ring in rings.values():
            ring.close()

        # Notify parent that process is shutting down
        try:
            ctrl_send.send(("shutdown", exit_code))
        except (BrokenPipeError, EOFError):
            pass
