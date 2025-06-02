# kmv/ui/chrome/telemetry.py
from __future__ import annotations
from typing import Mapping, Any

from PySide6.QtCore import Qt, QAbstractTableModel, QModelIndex

class TelemetryModel(QAbstractTableModel):
    """Simple 2-column table: | Metric | Value |."""
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._rows: list[tuple[str, Any]] = []

    # ── Qt API ────────────────────────────────────────────────────────────
    def rowCount(self, *_):          # type: ignore[override]
        return len(self._rows)

    def columnCount(self, *_):       # type: ignore[override]
        return 2

    def data(self, index: QModelIndex, role=Qt.DisplayRole):   # type: ignore[override]
        if role != Qt.DisplayRole:
            return None
        key, val = self._rows[index.row()]
        return key if index.column() == 0 else val

    # ── public API ────────────────────────────────────────────────────────
    def update(self, metrics: Mapping[str, Any]) -> None:
        """
        Replace all rows in one shot.  Cheap for ≤ few-hundred metrics.
        """
        self.beginResetModel()
        self._rows = sorted(metrics.items())        # keep table stable
        self.endResetModel()
