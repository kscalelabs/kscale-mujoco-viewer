"""Real-time multi-curve plot widget."""

from collections import deque
from typing import Mapping

import pyqtgraph as pg
from PySide6.QtWidgets import QVBoxLayout, QWidget


class ScalarPlot(QWidget):
    """Light-weight scrolling plot for a handful of scalar streams."""

    def __init__(
        self,
        *,
        history: int = 1_00,
        max_curves: int = 32,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._history = history
        self._max_curves = max_curves

        # --- graphics layout: [ plot | legend ] --------------------------
        layout = QVBoxLayout(self)

        self._glw  = pg.GraphicsLayoutWidget()
        layout.addWidget(self._glw)

        self._plot = self._glw.addPlot(row=0, col=0)
        self._plot.setClipToView(True)
        self._plot.showGrid(x=True, y=True)

        self._legend = pg.LegendItem(
            colCount=1,
            pen=pg.mkPen('#AAAAAA'),
            brush=pg.mkBrush(0, 0, 0, 150),
        )
        self._glw.addItem(self._legend, row=0, col=1)   # <-- directly in col 1

        # make the data column elastic and the legend column "shrink-to-fit"
        grid = self._glw.ci.layout                    # QGraphicsGridLayout
        grid.setColumnStretchFactor(0, 1)             # plot stretches
        grid.setColumnStretchFactor(1, 0)             # legend = minimum size

        self._curves: dict[str, pg.PlotDataItem] = {}
        self._buffers: dict[str, deque[tuple[float, float]]] = {}

        self._palette = [
            "#FF6B6B",
            "#4ECDC4",
            "#45B7D1",
            "#96CEB4",
            "#FFEAA7",
            "#DDA0DD",
            "#FF8C42",
            "#98D8C8",
            "#F7DC6F",
            "#BB8FCE",
            "#85C1E9",
            "#F8C471",
            "#82E0AA",
            "#F1948A",
            "#AED6F1",
            "#D7DBDD",
        ]
        self._color_index = 0

    def _next_color(self) -> str:
        color = self._palette[self._color_index % len(self._palette)]
        self._color_index += 1
        return color

    def update_data(self, t: float, scalars: Mapping[str, float]) -> None:
        """Append one `(t, value)` sample per scalar and redraw curves."""
        for name, value in scalars.items():
            if name not in self._buffers:
                if len(self._curves) >= self._max_curves:
                    continue  # silently ignore extra streams
                self._buffers[name] = deque(maxlen=self._history)
                color = self._next_color()
                curve = self._plot.plot(pen=pg.mkPen(color=color, width=2), name=name)
                self._curves[name] = curve
                self._legend.addItem(curve, name)      # manual, now that legend is external

            self._buffers[name].append((t, value))

        # Update the curves
        for name, buf in self._buffers.items():
            ts, vs = zip(*buf)
            self._curves[name].setData(ts, vs)
