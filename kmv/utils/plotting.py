"""Multi-plot dock widgets for live visualization."""

from collections import deque
from PySide6.QtCore import Slot
from PySide6.QtWidgets import QDockWidget, QWidget, QVBoxLayout
import pyqtgraph as pg
import mujoco



class BasePlotDock(QDockWidget):
    """Generic multi-plot dock, subclass to add curves."""

    DEFAULT_WIDTH = 250  # px

    def __init__(self, title: str, parent=None):
        super().__init__(title, parent)
        self._layout = pg.GraphicsLayoutWidget()        # a grid container
        self.setWidget(self._layout)
        self._curves: dict[str, pg.PlotDataItem] = {}   # name -> curve
        self.create_plots()


    def create_plots(self) -> None:
        """Add plot items & curves to self._layout."""
        raise NotImplementedError


    def _add_plot(self, y_label: str, name: str, pen="y") -> pg.PlotDataItem:
        plt = self._layout.addPlot(row=len(self._curves), col=0)
        plt.setLabel("left", y_label)
        plt.setLabel("bottom", "time (s)")
        curve = plt.plot(pen=pen)
        self._curves[name] = curve
        return curve

    def reset(self) -> None:
        for c in self._curves.values():
            c.setData([], [])


class PhysicsPlotsDock(BasePlotDock):
    """One plot per DOF: left = qpos[i], right = qvel[i]."""

    MAX_DOFS   = 5              # tweak as needed
    HISTORY    = 5_000           # points kept
    WINDOW_SEC = 5.0        # show last 5 s only
    POS_COLOR  = "y"
    VEL_COLOR  = "g"

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, *a, **kw):
        self._model = model
        self._data  = data
        self._dofs  = min(model.nq, model.nv, self.MAX_DOFS)

        self._x  = deque(maxlen=self.HISTORY)
        # _y["qpos"][i] is deque for qpos[i]; same for qvel
        self._y  = {
            "qpos": [deque(maxlen=self.HISTORY) for _ in range(self._dofs)],
            "qvel": [deque(maxlen=self.HISTORY) for _ in range(self._dofs)],
        }
        super().__init__("Mujoco physics plots", *a, **kw)

    def create_plots(self) -> None:
        self._viewboxes = []          # keep handles so we can slide them fast
        for idx in range(self._dofs):
            # qpos plot (left column)
            qpos_name = f"qpos[{idx}]"
            self._layout.addLabel(qpos_name, row=idx, col=0)  # tiny label
            plt_pos = self._layout.addPlot(row=idx, col=1)
            plt_pos.setLabel("left", qpos_name)
            vb_pos  = plt_pos.getViewBox()
            vb_pos.setXRange(0, self.WINDOW_SEC, padding=0)
            vb_pos.enableAutoRange(y=True, x=False)     # y still auto, x fixed
            curve_pos = plt_pos.plot(pen=self.POS_COLOR)
            self._viewboxes.append(vb_pos)

            # qvel plot (right column)
            qvel_name = f"qvel[{idx}]"
            self._layout.addLabel(qvel_name, row=idx, col=2)
            plt_vel = self._layout.addPlot(row=idx, col=3)
            plt_vel.setLabel("left", qvel_name)
            vb_vel  = plt_vel.getViewBox()
            vb_vel.setXRange(0, self.WINDOW_SEC, padding=0)
            vb_vel.enableAutoRange(y=True, x=False)
            curve_vel = plt_vel.plot(pen=self.VEL_COLOR)
            self._viewboxes.append(vb_vel)

            # store handles
            self._curves.setdefault("qpos", []).append(curve_pos)
            self._curves.setdefault("qvel", []).append(curve_vel)

        # stretch last column so plots fill horizontally
        self._layout.ci.layout.setColumnStretchFactor(3, 2)

    @Slot(float)
    def on_step(self, total_time: float) -> None:
        self._x.append(total_time)

        for i in range(self._dofs):
            # push new data
            self._y["qpos"][i].append(float(self._data.qpos[i]))
            self._y["qvel"][i].append(float(self._data.qvel[i]))
            # update curves
            self._curves["qpos"][i].setData(self._x, self._y["qpos"][i])
            self._curves["qvel"][i].setData(self._x, self._y["qvel"][i])

        # after all curves updated, slide every viewbox
        left = total_time - self.WINDOW_SEC
        right = total_time
        for vb in self._viewboxes:
            vb.setXRange(left, right, padding=0)     # instant visual pan

    def reset(self) -> None:
        self._x.clear()
        for lst in self._y.values():
            for dq in lst:
                dq.clear()
        for curve_lists in self._curves.values():
            for c in curve_lists:
                c.clear()
                c.getViewBox().enableAutoRange(x=True, y=True)

    def set_mjdata(self, data: mujoco.MjData) -> None:
        self._data = data

class ScalarPlotsDock(BasePlotDock):
    """
    Live plots for arbitrary scalar time-series.

    Usage
    -----
        dock.push(t, {"reward_total": 1.23, "upright": 0.45})

    A curve is created the first time a key appears.
    """

    HISTORY    = 5_000
    WINDOW_SEC = 5.0

    def __init__(self, title="Scalar plots", parent=None):
        self._x   = deque(maxlen=self.HISTORY)
        self._y   = {}            # key -> deque[float]
        self._viewboxes = []      # for fast sliding
        super().__init__(title, parent)

    # BasePlotDock calls this once in __init__.  We start empty.
    def create_plots(self) -> None:            # noqa: D401
        pass

    def push(self, t: float, scalars: dict[str, float]) -> None:
        """Append one time-stamp worth of scalars."""
        self._x.append(t)
        for key, val in scalars.items():
            if key not in self._y:                 # first sight → add curve
                self._y[key] = deque(maxlen=self.HISTORY)
                self._add_curve(key)
            self._y[key].append(val)
            self._curves[key].setData(self._x, self._y[key])

        # slide window
        left  = t - self.WINDOW_SEC
        right = t
        for vb in self._viewboxes:
            vb.setXRange(left, right, padding=0)

    def _add_curve(self, name: str) -> None:
        row = len(self._curves)
        plt = self._layout.addPlot(row=row, col=0)
        plt.setLabel("left", name)
        plt.setLabel("bottom", "time (s)")
        vb = plt.getViewBox()
        vb.setXRange(0, self.WINDOW_SEC, padding=0)
        vb.enableAutoRange(y=True, x=False)
        curve = plt.plot()      # let pyqtgraph pick a colour

        self._viewboxes.append(vb)
        self._curves[name] = curve

    def reset(self) -> None:
        """Clear all data and curves."""
        self._x.clear()
        for dq in self._y.values():
            dq.clear()
        for curve in self._curves.values():
            curve.clear()
            curve.getViewBox().enableAutoRange(x=True, y=True)