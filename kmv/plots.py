"""Very small dock widget that live-plots the first generalized position (qpos[0])."""

from collections import deque
from PySide6.QtCore import Slot
from PySide6.QtWidgets import QDockWidget, QWidget, QVBoxLayout
import pyqtgraph as pg
import mujoco


class QPosPlot(QDockWidget):
    """Right-hand dock that keeps a scrolling plot of qpos[0] vs. simulation time."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        history: int = 5_000,          # keep last N samples
        parent: QWidget | None = None,
    ) -> None:
        super().__init__("qpos[0]", parent)

        self._data = data
        self._x = deque(maxlen=history)
        self._y = deque(maxlen=history)

        central = QWidget(self)
        layout = QVBoxLayout(central)
        self._plot = pg.PlotWidget()
        self._plot.setLabel("left", text="qpos[0] (rad|m)")
        self._plot.setLabel("bottom", text="time (s)")
        self._curve = self._plot.plot(pen="y")
        layout.addWidget(self._plot)
        self.setWidget(central)

    # ------------------------------------------------------------------ #
    # Slot: called each time the physics engine signals a step
    # ------------------------------------------------------------------ #
    @Slot(float)
    def on_step(self, sim_time: float) -> None:
        self._x.append(sim_time)
        self._y.append(float(self._data.qpos[0]))
        self._curve.setData(self._x, self._y)


    # inside class QPosPlot

    def reset(self) -> None:
        """Clear plot data."""
        self._x.clear()
        self._y.clear()
        self._curve.clear()