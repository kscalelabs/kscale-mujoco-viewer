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
from kmv.core.types import ViewerConfig

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
    QLabel,
)
# QAction actually sits in QtGui
from PySide6.QtGui import QAction

from kmv.ipc.state       import SharedArrayRing
from kmv.ui.viewport     import GLViewport
from kmv.ui.plot         import ScalarPlot
from kmv.ui.table        import ViewerStatsTable


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
        view_conf: ViewerConfig,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        cfg = view_conf

        width, height   = cfg.width, cfg.height
        enable_plots    = cfg.enable_plots

        shadow          = cfg.shadow
        reflection      = cfg.reflection
        contact_force   = cfg.contact_force
        contact_point   = cfg.contact_point
        inertia         = cfg.inertia

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
        bar = QStatusBar(self)
        # add 8-pixel padding on the left (tweak the number to taste)
        bar.setContentsMargins(16, 0, 0, 0)
        bar.setSizeGripEnabled(False)
        self.setStatusBar(bar)

        def _add_status(label: str) -> QLabel:
            w = QLabel(label, self)
            w.setMinimumWidth(96)          # stable layout
            bar.addWidget(w)
            return w

        self._lbl_fps    = _add_status("FPS: –")
        self._lbl_phys   = _add_status("Phys Iters/s: –")
        self._lbl_simt   = _add_status("Sim Time: –")
        self._lbl_wallt  = _add_status("Wall Time: –")
        self._lbl_reset  = _add_status("Resets: 0")

        # -- live scalar plot -------------------------------------------------- #
        self._plots: dict[str, ScalarPlot] = {}
        self._plot_docks:  dict[str, QDockWidget] = {}
        self._plot_actions: dict[str, QAction] = {}
        self._enable_plots = enable_plots

        # ── menu bar → "Plots" drop-down ---------------------------------- #
        menubar = self.menuBar()
        menubar.setNativeMenuBar(False)      # ← keep it inside the window on macOS
        self._plots_menu = menubar.addMenu("&Plots")

        # ------------------------------------------------------------------ #
        #  NEW  menu just for the Telemetry table
        # ------------------------------------------------------------------ #
        self._telemetry_menu = menubar.addMenu("&Viewer Stats")

        # -- telemetry table --------------------------------------------------- #
        self._telemetry_table = ViewerStatsTable(self)      # <-- widget, not model
        table_dock = QDockWidget("Viewer Stats", self)
        table_dock.setWidget(self._telemetry_table)
        self.addDockWidget(Qt.RightDockWidgetArea, table_dock)
        table_dock.hide()                                 # start hidden

        #   menu ↔ dock synchronisation (checkbox behaviour)
        telem_action = QAction("Show viewer stats", self, checkable=True)
        telem_action.setChecked(False)                    # unchecked = hidden
        telem_action.toggled.connect(table_dock.setVisible)
        table_dock.visibilityChanged.connect(telem_action.setChecked)
        self._telemetry_menu.addAction(telem_action)

        self.show()

        # ── absolute-time bookkeeping -------------------------------- #
        self._sim_prev      = 0.0          # previous sim_time sample
        self._sim_offset    = 0.0          # accumulated time before last reset
        self._abs_sim_time  = 0.0          # latest absolute sim time
        self._reset_tol     = 1e-9         # tiny epsilon to ignore FP jitter

        # ── apply *initial camera* parameters (optional) ------------------ #
        cam = self._viewport.cam
        if cfg.camera_distance  is not None: cam.distance  = cfg.camera_distance
        if cfg.camera_azimuth   is not None: cam.azimuth   = cfg.camera_azimuth
        if cfg.camera_elevation is not None: cam.elevation = cfg.camera_elevation
        if cfg.camera_lookat    is not None: cam.lookat[:] = np.asarray(cfg.camera_lookat, dtype=np.float64)
        if cfg.track_body_id    is not None:
            cam.trackbodyid = cfg.track_body_id
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
        """
        Lazily create one dock + QAction per *group*.  The dock starts hidden;
        ticking the corresponding check-box in the **Plots** menu shows it.
        """
        if group in self._plots:
            return self._plots[group]

        # (1) graphics widget ------------------------------------------------ #
        plot = ScalarPlot(history=600, max_curves=32)
        dock = QDockWidget(group.capitalize(), self)
        dock.setWidget(plot)
        self.addDockWidget(Qt.BottomDockWidgetArea, dock)
        dock.hide()                                    # ← hidden by default

        # (2) menu action ---------------------------------------------------- #
        action = QAction(group.capitalize(), self, checkable=True)
        action.setChecked(False)                      # unchecked = hidden
        # bidirectional sync  (menu → dock)
        action.toggled.connect(dock.setVisible)
        # …and (dock → menu) in case user closes the dock title-bar "X"
        dock.visibilityChanged.connect(action.setChecked)
        self._plots_menu.addAction(action)

        # (3) bookkeeping ---------------------------------------------------- #
        self._plots[group]        = plot
        self._plot_docks[group]   = dock
        self._plot_actions[group] = action
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
            if "Phys Iters" in msg:
                phys_iters_value = int(msg["Phys Iters"])

        # -- 2b. compute phys iters / sec ----------------------------------- #
        if phys_iters_value is not None:
            now = time.perf_counter()
            dt  = now - self._phys_iters_prev_time
            if dt > 0:
                self._phys_iters_per_sec = (phys_iters_value - self._phys_iters_prev) / dt
            self._phys_iters_prev      = phys_iters_value
            self._phys_iters_prev_time = now

        # -- 2c. GUI-local metrics ------------------------------------------ #
        rows["Viewer FPS"]        = round(self._fps, 1)
        rows["Plot FPS"]    = round(self._plot_hz, 1)
        rows["Phys Iters/s"]   = round(self._phys_iters_per_sec, 1)
        rows["Abs Sim Time"]  = round(self._abs_sim_time, 3)
        rows["Sim Time"]      = round(sim_time, 3)              # NEW
        rows["Wall Time"]     = round(wall_elapsed, 2)          # NEW
        rows["Reset Count"]     = self._reset_count               # NEW
        rows["Sim Time / Real Time"] = round(realtime_x, 2)           # NEW

        self._telemetry_table.update(rows)

        # ------------------------------------------------------------------ #
        #  Status-bar text refresh (cheap – a few QString ops per frame)
        # ------------------------------------------------------------------ #
        self._lbl_fps.setText(f"FPS: {self._fps:5.1f}")
        self._lbl_phys.setText(f"Phys/s: {self._phys_iters_per_sec:5.1f}")
        self._lbl_simt.setText(f"Sim t: {sim_time:6.2f}")
        self._lbl_wallt.setText(f"Wall t: {wall_elapsed:6.2f}")
        self._lbl_reset.setText(f"Resets: {self._reset_count}")

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
