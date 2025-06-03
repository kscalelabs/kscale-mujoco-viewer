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
        self._plots: dict[str, ScalarPlot] = {}
        self._enable_plots = enable_plots

        # -- telemetry table --------------------------------------------------- #
        self._telemetry_table = TelemetryTable(self)      # <-- widget, not model
        table_dock = QDockWidget("Telemetry", self)
        table_dock.setWidget(self._telemetry_table)
        self.addDockWidget(Qt.RightDockWidgetArea, table_dock)

        self.show()

        # ── absolute-time bookkeeping -------------------------------- #
        self._sim_prev      = 0.0          # previous sim_time sample
        self._sim_offset    = 0.0          # accumulated time before last reset
        self._abs_sim_time  = 0.0          # latest absolute sim time
        self._reset_tol     = 1e-9         # tiny epsilon to ignore FP jitter

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

        # ── physics-state push throughput ----------------------------------- #
        self._phys_iters_prev      = 0
        self._phys_iters_prev_time = time.perf_counter()
        self._phys_iters_per_sec   = 0.0

        # ── NEW: wall-clock & reset tracking -------------------------------- #
        self._wall_start   : float | None = None   # set on first frame
        self._reset_count  = 0                    # increments on every sim reset

    # ───────────────────────────────────────────────────────────────────── #
    def _plot_for_group(self, group: str) -> ScalarPlot:
        if group in self._plots:
            return self._plots[group]

        plot = ScalarPlot(history=600, max_curves=32)
        dock = QDockWidget(group.capitalize(), self)
        dock.setWidget(plot)
        self.addDockWidget(Qt.BottomDockWidgetArea, dock)
        self._plots[group] = plot
        return plot

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

        # ---- absolute sim time (handles resets) --------------------- #
        if sim_time < self._sim_prev - self._reset_tol:      # reset detected
            self._sim_offset += self._sim_prev
            self._reset_count += 1                           # ← NEW counter
        self._sim_prev     = sim_time
        self._abs_sim_time = self._sim_offset + sim_time

        # ---- wall-clock bookkeeping --------------------------------- #
        if self._wall_start is None:                         # first frame
            self._wall_start = now_gui
        wall_elapsed = now_gui - self._wall_start
        realtime_x   = (
            self._abs_sim_time / max(wall_elapsed, 1e-9)
            if wall_elapsed > 0.0 else 0.0
        )

        self._data.qpos[:] = qpos
        self._data.qvel[:] = qvel
        self._data.time = sim_time
        mujoco.mj_forward(self._model, self._data)

        # -- 2a. pull parent-provided rows ----------------------------------- #
        rows: dict[str, float] = {}
        phys_iters_value: int | None = None

        while not self._table_q.empty():
            msg = self._table_q.get_nowait()
            rows.update(msg)
            if "phys iters" in msg:
                phys_iters_value = int(msg["phys iters"])

        # -- 2b. compute phys iters / sec ----------------------------------- #
        if phys_iters_value is not None:
            now = time.perf_counter()
            dt  = now - self._phys_iters_prev_time
            if dt > 0:
                self._phys_iters_per_sec = (phys_iters_value - self._phys_iters_prev) / dt
            self._phys_iters_prev      = phys_iters_value
            self._phys_iters_prev_time = now

        # -- 2c. GUI-local metrics ------------------------------------------ #
        rows["FPS"]        = round(self._fps, 1)
        rows["plot Hz"]    = round(self._plot_hz, 1)
        rows["phys iters/s"]   = round(self._phys_iters_per_sec, 1)
        rows["abs sim t"]  = round(self._abs_sim_time, 3)
        rows["sim t"]      = round(sim_time, 3)              # NEW
        rows["wall t"]     = round(wall_elapsed, 2)          # NEW
        rows["resets"]     = self._reset_count               # NEW
        rows["× real-time"] = round(realtime_x, 2)           # NEW

        self._telemetry_table.update(rows)

        # -- 2b. update scalar plots (one panel per group) --------------- #
        if self._enable_plots:
            n_drained = 0
            while not self._plot_q.empty():
                msg      = self._plot_q.get_nowait()
                group    = msg.get("group", "default")
                scalars  = msg["scalars"]
                plot     = self._plot_for_group(group)
                plot.update_data(self._abs_sim_time, scalars)
                n_drained += 1
            self._plot_ctr += n_drained

        if (now - self._plot_timer) >= 1.0:
            self._plot_hz   = self._plot_ctr / (now - self._plot_timer)
            self._plot_ctr  = 0
            self._plot_timer = now

        # -- 3. repaint ------------------------------------------------------ #
        self._viewport.update()   # Qt will schedule paintGL()
