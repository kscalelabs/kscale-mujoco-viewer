"""Real-time multi-curve plot widget."""

from collections import deque
from typing import Mapping

import pyqtgraph as pg
from PySide6.QtWidgets import QSizePolicy, QVBoxLayout, QWidget

POL = QSizePolicy.Policy


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

        # Make layout widget
        layout = QVBoxLayout(self)
        self._glw = pg.GraphicsLayoutWidget()
        layout.addWidget(self._glw)

        # Add plot
        self._plot = self._glw.addPlot(row=0, col=0)
        self._plot.setClipToView(True)
        self._plot.showGrid(x=True, y=True)

        # Add legend
        self._legend = pg.LegendItem(
            colCount=1,
            pen=None,
            brush=pg.mkBrush(0, 0, 0, 150),
            verSpacing=0,
            horSpacing=0,
        )
        self._glw.addItem(self._legend, row=0, col=1)

        # Adjust how legend is displayed
        self._legend.setSizePolicy(POL.Preferred, POL.Fixed)
        self._legend.updateSize()
        h = self._legend.boundingRect().height()
        self._legend.setMaximumHeight(h)
        grid = self._glw.ci.layout
        grid.setColumnStretchFactor(0, 1)
        grid.setColumnStretchFactor(1, 0)

        # Initialize buffers and curves
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

                # Add new curve to plot
                self._buffers[name] = deque(maxlen=self._history)
                color = self._next_color()
                curve = self._plot.plot(pen=pg.mkPen(color=color, width=2), name=name)
                self._curves[name] = curve

                # Add new curve to legend
                self._legend.addItem(curve, name)

            self._buffers[name].append((t, value))

        # Update the curves
        for name, buf in self._buffers.items():
            ts, vs = zip(*buf)
            self._curves[name].setData(ts, vs)
