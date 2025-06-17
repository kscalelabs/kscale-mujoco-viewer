"""Qt front-end window.

Hosts an OpenGL viewport, optional scalar plots, and a live telemetry table,
fed by shared-memory rings and metric queues from the parent process.
"""

import queue
from multiprocessing import Queue
from multiprocessing.connection import Connection
from typing import Mapping

import mujoco
import numpy as np
from PySide6.QtCore import Qt

# QAction actually sits in QtGui
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QDockWidget,
    QLabel,
    QMainWindow,
    QMenu,
    QStatusBar,
    QWidget,
)

from kmv.core.controller import RenderLoop
from kmv.core.types import PlotPacket, TelemetryPacket, ViewerConfig
from kmv.ipc.shared_ring import SharedMemoryRing
from kmv.ui.help import HelpWidget
from kmv.ui.plot import ScalarPlot
from kmv.ui.settings import SettingsWidget
from kmv.ui.table import ViewerStatsTable
from kmv.ui.viewport import GLViewport


class ViewerWindow(QMainWindow):
    """Main KMV GUI window composed of viewport, plots and stats."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        rings: Mapping[str, SharedMemoryRing],
        *,
        table_q: Queue,
        plot_q: Queue,
        ctrl_send: Connection,  # NEW
        view_conf: ViewerConfig,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        cfg = view_conf
        self.resize(cfg.width, cfg.height)
        self.setWindowTitle("K-Scale MuJoCo Viewer")

        self._model, self._data = model, data
        self._rings = rings
        self._table_q, self._plot_q = table_q, plot_q
        self._ctrl_send = ctrl_send
        self._enable_plots = cfg.enable_plots

        # MuJoCo scene
        self._viewport = GLViewport(
            model,
            data,
            shadow=cfg.shadow,
            reflection=cfg.reflection,
            contact_force=cfg.contact_force,
            contact_point=cfg.contact_point,
            inertia=cfg.inertia,
            on_forces=lambda arr: ctrl_send.send(("forces", arr)),
            parent=self,
        )
        self.setCentralWidget(self._viewport)

        # Status bar
        bar = QStatusBar(self)
        bar.setContentsMargins(16, 0, 0, 0)
        bar.setSizeGripEnabled(False)
        self.setStatusBar(bar)

        def _add_status(label: str) -> QLabel:
            w = QLabel(label, self)
            w.setMinimumWidth(96)
            bar.addWidget(w)
            return w

        self._lbl_fps = _add_status("FPS: –")
        self._lbl_phys = _add_status("Phys Iters/s: –")
        self._lbl_simt = _add_status("Sim Time: –")
        self._lbl_wallt = _add_status("Wall Time: –")
        self._lbl_reset = _add_status("Resets: 0")

        # Menus
        menubar = self.menuBar()
        menubar.setNativeMenuBar(False)
        self._plots_menu = menubar.addMenu("&Plots")
        self._plot_submenus: dict[str, QMenu] = {}
        self._telemetry_menu = menubar.addMenu("&Viewer Stats")
        self._settings_menu = menubar.addMenu("&Settings")
        self._help_menu = menubar.addMenu("&Help")

        # Viewer stats table
        self._viewer_stats_table = ViewerStatsTable(self)
        table_dock = QDockWidget("Viewer Stats", self)
        table_dock.setWidget(self._viewer_stats_table)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, table_dock)
        table_dock.hide()

        telem_action = QAction("Show viewer stats", self, checkable=True)
        telem_action.toggled.connect(table_dock.setVisible)
        table_dock.visibilityChanged.connect(telem_action.setChecked)
        self._telemetry_menu.addAction(telem_action)

        # Settings panel
        def _set_vis_flag(flag: int, state: bool) -> None:
            self._viewport.opt.flags[flag] = state
            self._viewport.update()

        def _set_opt_attr(attr: str, value: int) -> None:
            setattr(self._viewport.opt, attr, value)
            self._viewport.update()

        settings_widget = SettingsWidget(
            get_set_flag=_set_vis_flag,
            set_opt_attr=_set_opt_attr,
            force_init=cfg.contact_force,
            point_init=cfg.contact_point,
            inertia_init=cfg.inertia,
            joint_init=False,
            label_init=self._viewport.opt.label,
            frame_init=self._viewport.opt.frame,
            parent=self,
        )
        settings_dock = QDockWidget("Settings", self)
        settings_dock.setWidget(settings_widget)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, settings_dock)
        settings_dock.hide()

        settings_action = QAction("Show settings", self, checkable=True)
        settings_action.toggled.connect(settings_dock.setVisible)
        settings_dock.visibilityChanged.connect(settings_action.setChecked)
        self._settings_menu.addAction(settings_action)

        # Help widget
        self._help_widget = HelpWidget(self)
        help_dock = QDockWidget("Help", self)
        help_dock.setWidget(self._help_widget)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, help_dock)
        help_dock.hide()

        help_action = QAction("Show help", self, checkable=True)
        help_action.toggled.connect(help_dock.setVisible)
        help_dock.visibilityChanged.connect(help_action.setChecked)
        self._help_menu.addAction(help_action)

        # Plots
        self._plots: dict[str, ScalarPlot] = {}
        self._plot_docks: dict[str, QDockWidget] = {}
        self._plot_actions: dict[str, QAction] = {}

        # Camera
        cam = self._viewport.cam
        if cfg.camera_distance is not None:
            cam.distance = cfg.camera_distance
        if cfg.camera_azimuth is not None:
            cam.azimuth = cfg.camera_azimuth
        if cfg.camera_elevation is not None:
            cam.elevation = cfg.camera_elevation
        if cfg.camera_lookat is not None:
            cam.lookat[:] = np.asarray(cfg.camera_lookat, dtype=np.float64)
        if cfg.track_body_id is not None:
            cam.trackbodyid = cfg.track_body_id
            cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING

        self.__post_init_render_loop()
        self.show()

    def __post_init_render_loop(self) -> None:
        """Wire shared queues into a `RenderLoop` instance."""

        def _pop(q: Queue) -> object | None:
            try:
                return q.get_nowait()
            except queue.Empty:
                return None

        def get_table_packet() -> TelemetryPacket | None:
            msg = _pop(self._table_q)
            if isinstance(msg, dict):
                return TelemetryPacket(rows=msg)
            return None

        def get_plot_packet() -> PlotPacket | None:
            msg = _pop(self._plot_q)
            if isinstance(msg, dict) and "scalars" in msg:
                return PlotPacket(group=msg.get("group", "default"), scalars=msg["scalars"])
            return None

        self._rl = RenderLoop(
            model=self._model,
            data=self._data,
            rings=self._rings,
            on_forces=lambda a: self._ctrl_send.send(("forces", a)),
            get_table=get_table_packet,
            get_plot=get_plot_packet,
        )

    def _menu_for_path(self, path: str) -> tuple[QMenu, str]:
        """Return (parent_menu, leaf_name) for a slash-separated group path."""
        parts = path.split("/")
        parent: QMenu = self._plots_menu  # "Plots" menu is the root
        sofar = ""

        # Walk all but the last component, creating sub-menus as needed
        for comp in parts[:-1]:
            sofar = f"{sofar}/{comp}" if sofar else comp
            if sofar not in self._plot_submenus:
                submenu = parent.addMenu(comp.capitalize())
                self._plot_submenus[sofar] = submenu
            parent = self._plot_submenus[sofar]

        return parent, parts[-1]

    def _plot_for_group(self, group: str) -> ScalarPlot:
        """Return (or lazily create) the plot dock for *group*."""
        if group in self._plots:
            return self._plots[group]

        # Get the parent menu and the leaf name
        parent_menu, leaf_name = self._menu_for_path(group)

        # Graphics widget
        plot = ScalarPlot(history=600, max_curves=32)
        dock = QDockWidget(leaf_name.capitalize(), self)
        dock.setWidget(plot)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)
        dock.hide()

        # Menu action
        action = QAction(leaf_name.capitalize(), self, checkable=True)
        action.setChecked(False)
        # Bidirectional sync (menu to dock)
        action.toggled.connect(dock.setVisible)
        # …and (dock to menu) in case user closes the dock title-bar "X"
        dock.visibilityChanged.connect(action.setChecked)
        parent_menu.addAction(action)

        # Bookkeeping
        self._plots[group] = plot
        self._plot_docks[group] = dock
        self._plot_actions[group] = action
        return plot

    def step_and_draw(self) -> None:
        """Advance one GUI frame: pull state, update widgets, repaint."""
        self._rl.tick()

        # Table
        self._viewer_stats_table.refresh(self._rl._last_table)

        # Status-bar mirrors
        self._lbl_fps.setText(f"FPS: {self._rl.fps:5.1f}")
        self._lbl_phys.setText(f"Phys/s: {self._rl.phys_iters_per_sec:5.1f}")
        self._lbl_simt.setText(f"Sim t: {self._rl.sim_time_abs:6.2f}")
        self._lbl_wallt.setText(f"Wall t: {self._rl._last_table.get('Wall Time', 0):6.2f}")
        self._lbl_reset.setText(f"Resets: {self._rl.reset_count}")

        # Plots
        if self._enable_plots:
            for group, scalars in self._rl._plots_latest.items():
                plot = self._plot_for_group(group)
                plot.update_data(self._rl.sim_time_abs, scalars)

        # Repaint OpenGL
        self._viewport.update()
