# kmv/worker/window.py
"""
`ViewerWindow` – Qt front-end that lives **inside** the GUI process.

It receives:
* a compiled `mjModel` and private `mjData`
* a dict of SharedArrayRings (state streams)
* the telemetry queue
* a few cosmetic options

The parent process never imports this module.
"""

from __future__ import annotations

import time
from typing import Mapping

import mujoco
import numpy as np

from PySide6.QtCore    import Qt
from PySide6.QtWidgets import (
    QMainWindow,
    QDockWidget,
    QStatusBar,
    QWidget,
    QVBoxLayout,
    QTableView,
)

from kmv.ipc.state       import SharedArrayRing
from kmv.ui.viewport     import GLViewport
from kmv.ui.plot         import ScalarPlot
from kmv.ui.table        import TelemetryTable


class ViewerWindow(QMainWindow):
    """
    Composes:  | GLViewport |  ScalarPlot  |  TelemetryTable |
    """

    # ------------------------------------------------------------------ #

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        rings: Mapping[str, SharedArrayRing],
        *,
        table_q,
        plot_q,
        view_opts: dict[str, object] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        opts = view_opts or {}

        # ── generic opts -------------------------------------------------- #
        width         = int(opts.get("width", 900))
        height        = int(opts.get("height", 550))
        enable_plots  = bool(opts.get("enable_plots", True))

        # ── visual flags -------------------------------------------------- #
        shadow        = bool(opts.get("shadow",        False))
        reflection    = bool(opts.get("reflection",    False))
        contact_force = bool(opts.get("contact_force", False))
        contact_point = bool(opts.get("contact_point", False))
        inertia       = bool(opts.get("inertia",       False))

        self.resize(width, height)
        self.setWindowTitle("KMV Viewer")

        # -- keep references -------------------------------------------------- #
        self._model      = model
        self._data       = data
        self._rings      = rings
        self._table_q    = table_q
        self._plot_q     = plot_q

        # -- central OpenGL viewport ----------------------------------------- #
        self._viewport = GLViewport(
            model,
            data,
            shadow        = shadow,
            reflection    = reflection,
            contact_force = contact_force,
            contact_point = contact_point,
            inertia       = inertia,
            parent        = self,
        )
        self.setCentralWidget(self._viewport)

        # -- status bar ------------------------------------------------------- #
        self.setStatusBar(QStatusBar(self))

        # -- live scalar plot -------------------------------------------------- #
        if enable_plots:
            self._scalar_plot = ScalarPlot(history=600, max_curves=32)
            dock = QDockWidget("Scalars", self)
            dock.setWidget(self._scalar_plot)
            self.addDockWidget(Qt.BottomDockWidgetArea, dock)
        else:
            self._scalar_plot = None

        # -- telemetry table --------------------------------------------------- #
        self._telemetry_table = TelemetryTable(self)      # <-- widget, not model
        table_dock = QDockWidget("Telemetry", self)
        table_dock.setWidget(self._telemetry_table)
        self.addDockWidget(Qt.RightDockWidgetArea, table_dock)

        self.show()

        # ── apply *initial camera* parameters (optional) ------------------ #
        cam = self._viewport.cam
        if "camera_distance"  in opts: cam.distance  = float(opts["camera_distance"])
        if "camera_azimuth"   in opts: cam.azimuth   = float(opts["camera_azimuth"])
        if "camera_elevation" in opts: cam.elevation = float(opts["camera_elevation"])
        if "camera_lookat"    in opts:
            cam.lookat[:] = np.asarray(opts["camera_lookat"], dtype=np.float64)
        if opts.get("track_body_id") is not None:
            cam.trackbodyid = int(opts["track_body_id"])
            cam.type        = mujoco.mjtCamera.mjCAMERA_TRACKING

        # ── self-computed performance metrics ──────────────────────────────
        self._fps_timer    = time.perf_counter()
        self._frame_ctr    = 0
        self._fps          = 0.0          # latest 1-s average

        self._plot_timer   = time.perf_counter()
        self._plot_ctr     = 0
        self._plot_hz      = 0.0          # latest 1-s average

        # ── sim-loop throughput (computed from 'iters' field) ────────────────
        self._iters_prev      = 0
        self._iters_prev_time = time.perf_counter()
        self._iters_per_sec   = 0.0

    # ------------------------------------------------------------------ #
    #  Timer callback
    # ------------------------------------------------------------------ #

    def step_and_draw(self) -> None:
        """
        Called ~60 Hz by a `QTimer` in `worker.entrypoint.run_worker`.
        1. Pull newest qpos/qvel from shared memory.
        2. Run `mj_forward` so contacts are fresh for rendering.
        3. Drain metrics queue → update plot & table.
        4. Trigger an OpenGL repaint.
        """
        # GUI counter - lets you watch GUI cadence and instantaneous backlog
        now_gui = time.perf_counter()
        try:
            backlog = self._plot_q._unfinished_tasks  # cheap & safe; works on mac
        except AttributeError:
            backlog = -1

        print(f"[GUI] {now_gui:10.3f}  sim_t={self._rings['sim_time'].latest()[0]:6.2f}"
              f"  backlog={backlog:3}")

        # -- 1. shared-memory read ------------------------------------------ #
        self._frame_ctr += 1                         # count every repaint
        now = time.perf_counter()
        if (now - self._fps_timer) >= 1.0:
            self._fps       = self._frame_ctr / (now - self._fps_timer)
            self._frame_ctr = 0
            self._fps_timer = now

        qpos = self._rings["qpos"].latest()
        qvel = self._rings["qvel"].latest()
        sim_time = float(self._rings["sim_time"].latest()[0])

        self._data.qpos[:] = qpos
        self._data.qvel[:] = qvel
        self._data.time = sim_time
        mujoco.mj_forward(self._model, self._data)

        # -- 2a. pull parent-provided rows ----------------------------------- #
        rows: dict[str, float] = {}
        iters_value: int | None = None

        while not self._table_q.empty():
            msg = self._table_q.get_nowait()
            rows.update(msg)
            if "iters" in msg:
                iters_value = int(msg["iters"])

        # -- 2b. compute iters / sec (GUI-side) ------------------------------ #
        if iters_value is not None:
            now = time.perf_counter()
            dt  = now - self._iters_prev_time
            if dt > 0:                                # avoid div-zero on first frame
                self._iters_per_sec = (iters_value - self._iters_prev) / dt
            self._iters_prev      = iters_value
            self._iters_prev_time = now

        # -- 2c. GUI-local metrics ------------------------------------------ #
        rows["FPS"]        = round(self._fps, 1)
        rows["plot Hz"]    = round(self._plot_hz, 1)
        rows["iters/s"]    = round(self._iters_per_sec, 1)

        self._telemetry_table.update(rows)

        # -- 2b. update scalar plot ------------------------------------- #
        if self._scalar_plot is not None:
            n_drained = 0
            latest = None
            while not self._plot_q.empty():
                latest = self._plot_q.get_nowait()
                n_drained += 1
            if latest is not None:
                self._scalar_plot.update_data(sim_time, latest)
                self._plot_ctr += 1
            print(f"        drained={n_drained:3}")

        if (now - self._plot_timer) >= 1.0:
            self._plot_hz   = self._plot_ctr / (now - self._plot_timer)
            self._plot_ctr  = 0
            self._plot_timer = now

        # -- 3. repaint ------------------------------------------------------ #
        self._viewport.update()   # Qt will schedule paintGL()
