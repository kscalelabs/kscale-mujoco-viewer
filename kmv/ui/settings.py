"""Checkbox panel for run-time visual settings."""

from typing import Callable

import mujoco
from PySide6.QtWidgets import QCheckBox, QFormLayout, QWidget


class SettingsWidget(QWidget):
    """Manages the settings panel for the MuJoCo Viewer."""

    def __init__(
        self,
        *,
        get_set_flag: Callable[[int, bool], None],
        force_init: bool,
        point_init: bool,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        self._set_flag = get_set_flag

        self._chk_force = QCheckBox("Show Contact Forces")
        self._chk_force.setChecked(force_init)
        self._chk_force.toggled.connect(lambda b: self._set_flag(mujoco.mjtVisFlag.mjVIS_CONTACTFORCE, b))

        self._chk_point = QCheckBox("Show Contact Points")
        self._chk_point.setChecked(point_init)
        self._chk_point.toggled.connect(lambda b: self._set_flag(mujoco.mjtVisFlag.mjVIS_CONTACTPOINT, b))

        lay = QFormLayout(self)
        lay.addRow(self._chk_force)
        lay.addRow(self._chk_point)
