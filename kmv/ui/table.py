"""Simple two-column Qt table for live telemetry."""

from __future__ import annotations

from typing import Any, Mapping

from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt
from PySide6.QtWidgets import QTableView, QWidget


class _TableModel(QAbstractTableModel):
    """Table class for stats and telemetry.

    The table is updated with the new metrics, and any keys that are missing
    in the new metrics are filled with `None` so the row order remains stable.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._rows: list[tuple[str, Any]] = []
        self._known_keys: set[str] = set()

    def rowCount(self, *_) -> int:
        return len(self._rows)

    def columnCount(self, *_) -> int:
        return 2

    def data(self, index: QModelIndex, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        key, val = self._rows[index.row()]
        return key if index.column() == 0 else val

    def replace(self, metrics: Mapping[str, Any]) -> None:
        """Refresh the table with the new metrics."""
        self._known_keys.update(metrics.keys())

        rows: list[tuple[str, Any]] = []
        for key in sorted(self._known_keys):
            rows.append((key, metrics.get(key, None)))

        self.beginResetModel()
        self._rows = rows
        self.endResetModel()


class ViewerStatsTable(QTableView):
    """Helper that Wraps `_TableModel` and adds `.update()`."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._model = _TableModel(self)
        self.setModel(self._model)

        self.horizontalHeader().setStretchLastSection(True)
        self.verticalHeader().hide()
        self.setSelectionMode(QTableView.NoSelection)
        self.setEditTriggers(QTableView.NoEditTriggers)

    def update(self, metrics: Mapping[str, Any]) -> None:
        """Replace all rows with updated metrics."""
        self._model.replace(metrics)
