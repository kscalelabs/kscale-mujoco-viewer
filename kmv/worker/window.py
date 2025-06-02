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
        metrics_q,
        view_opts: dict[str, object] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        opts = view_opts or {}

        width  = int(opts.get("width", 900))
        height = int(opts.get("height", 550))
        enable_plots = bool(opts.get("enable_plots", True))

        self.resize(width, height)
        self.setWindowTitle("KMV Viewer")

        # -- keep references -------------------------------------------------- #
        self._model      = model
        self._data       = data
        self._rings      = rings
        self._metrics_q  = metrics_q

        # -- central OpenGL viewport ----------------------------------------- #
        self._viewport = GLViewport(model, data, parent=self)
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
        # -- 1. shared-memory read ------------------------------------------ #
        qpos = self._rings["qpos"].latest()
        qvel = self._rings["qvel"].latest()

        self._data.qpos[:] = qpos
        self._data.qvel[:] = qvel
        mujoco.mj_forward(self._model, self._data)

        # -- 2. metrics queue ------------------------------------------------ #
        while not self._metrics_q.empty():
            metrics = self._metrics_q.get_nowait()
            # update table
            self._telemetry_table.update(metrics)
            # update scalar plot (if enabled)
            if self._scalar_plot is not None:
                self._scalar_plot.update_data(float(self._data.time), metrics)

        # -- 3. repaint ------------------------------------------------------ #
        self._viewport.update()   # Qt will schedule paintGL()
