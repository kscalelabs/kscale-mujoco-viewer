"""Tiny real-time plot panel for streaming scalar values."""

from __future__ import annotations

from collections import deque
from typing import Dict, Tuple

import pyqtgraph as pg
from PySide6.QtWidgets import QWidget, QVBoxLayout


class ScalarPlot(QWidget):
    """
    Lightweight QWidget that plots up to *max_curves* time–series in real-time.
    Designed for ∼k Hz data rates without blocking the sim thread.
    """

    def __init__(self, *, history: int = 1_000, max_curves: int = 16, parent=None) -> None:
        super().__init__(parent)
        self._history = history
        self._max_curves = max_curves

        layout = QVBoxLayout(self)
        self._plot = pg.PlotWidget()
        self._plot.setClipToView(True)
        self._plot.showGrid(x=True, y=True)
        self._plot.addLegend(offset=(10, 10))
        layout.addWidget(self._plot)

        self._curves: Dict[str, pg.PlotDataItem] = {}
        self._buffers: Dict[str, deque[Tuple[float, float]]] = {}
        
        # Color palette for different curves - using a mix of vibrant and distinguishable colors
        self._color_palette = [
            "#FF6B6B",  # Red
            "#4ECDC4",  # Teal
            "#45B7D1",  # Blue
            "#96CEB4",  # Light Green
            "#FFEAA7",  # Yellow
            "#DDA0DD",  # Plum
            "#FF8C42",  # Orange
            "#98D8C8",  # Mint
            "#F7DC6F",  # Light Yellow
            "#BB8FCE",  # Light Purple
            "#85C1E9",  # Light Blue
            "#F8C471",  # Light Orange
            "#82E0AA",  # Light Green
            "#F1948A",  # Light Red
            "#AED6F1",  # Very Light Blue
            "#D7DBDD",  # Light Gray
        ]
        self._color_index = 0

    def _get_next_color(self) -> str:
        """Get the next color from the palette, cycling if needed."""
        color = self._color_palette[self._color_index % len(self._color_palette)]
        self._color_index += 1
        return color

    def update_data(self, t: float, scalars: Dict[str, float]) -> None:
        """Append one sample per *scalars* key and refresh the plot."""
        for name, val in scalars.items():
            if name not in self._buffers:
                if len(self._curves) >= self._max_curves:
                    continue
                self._buffers[name] = deque(maxlen=self._history)
                
                # Get a unique color for this curve
                color = self._get_next_color()
                self._curves[name] = self._plot.plot(
                    pen=pg.mkPen(color=color, width=2), name=name
                )

            self._buffers[name].append((t, val))

        for name, buf in self._buffers.items():
            ts, vs = zip(*buf)
            self._curves[name].setData(ts, vs)
