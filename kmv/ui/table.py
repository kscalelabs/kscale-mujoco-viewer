"""Simple two-column Qt table for live telemetry."""

from typing import Any, Mapping

from PySide6.QtCore import (
    QAbstractTableModel,
    QModelIndex,
    QPersistentModelIndex,
    Qt,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QTableView,
    QWidget,
)


class _TableModel(QAbstractTableModel):
    """Table class for stats and telemetry.

    The table is updated with the new metrics, and any keys that are missing
    in the new metrics are filled with `None` so the row order remains stable.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._rows: list[tuple[str, Any]] = []
        self._known_keys: set[str] = set()

    def rowCount(  # noqa: N802
        self,
        parent: QModelIndex | QPersistentModelIndex = QModelIndex(),
    ) -> int:
        return len(self._rows)

    def columnCount(  # noqa: N802
        self,
        parent: QModelIndex | QPersistentModelIndex = QModelIndex(),
    ) -> int:
        return 2

    def data(
        self,
        index: QModelIndex | QPersistentModelIndex,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> object | None:
        if role != Qt.ItemDataRole.DisplayRole:
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
    """Helper that Wraps `_TableModel` and adds `.refresh()`."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._model = _TableModel(self)
        self.setModel(self._model)

        self.horizontalHeader().setStretchLastSection(True)
        self.verticalHeader().hide()
        self.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)

    def refresh(self, metrics: Mapping[str, Any]) -> None:
        """Replace all rows with updated metrics."""
        self._model.replace(metrics)
