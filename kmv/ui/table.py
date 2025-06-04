# kmv/ui/table.py
"""
`TelemetryTable` – a two-column Qt model+view for live key-value metrics.

No MuJoCo, no multiprocessing.  Pure PySide6.
"""

from __future__ import annotations

from typing import Any, Mapping

from PySide6.QtCore    import Qt, QAbstractTableModel, QModelIndex
from PySide6.QtWidgets import QTableView


class _TableModel(QAbstractTableModel):
    """Internal two-column model:  | Metric | Value |"""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._rows: list[tuple[str, Any]] = []
        self._known_keys: set[str] = set()

    def rowCount(self, *_):          # type: ignore[override]
        return len(self._rows)

    def columnCount(self, *_):       # type: ignore[override]
        return 2

    def data(self, index: QModelIndex, role=Qt.DisplayRole):  # type: ignore[override]
        if role != Qt.DisplayRole:
            return None
        key, val = self._rows[index.row()]
        return key if index.column() == 0 else val

    def replace(self, metrics: Mapping[str, Any]) -> None:
        """
        Fast full refresh keeping *all* previously seen keys.

        Any key that is missing in *metrics* for this frame is filled with
        ``None`` so the row order remains stable.
        """
        self._known_keys.update(metrics.keys())

        rows: list[tuple[str, Any]] = []
        for key in sorted(self._known_keys):
            rows.append((key, metrics.get(key, None)))

        self.beginResetModel()
        self._rows = rows
        self.endResetModel()


class ViewerStatsTable(QTableView):
    """
    Thin wrapper that owns its `_TableModel` and exposes a single
    `.update(metrics: dict)` method.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._model = _TableModel(self)
        self.setModel(self._model)

        self.horizontalHeader().setStretchLastSection(True)
        self.verticalHeader().hide()
        self.setSelectionMode(QTableView.NoSelection)
        self.setEditTriggers(QTableView.NoEditTriggers)

    def update(self, metrics: Mapping[str, Any]) -> None:
        """Replace all rows with *metrics*."""
        self._model.replace(metrics)
