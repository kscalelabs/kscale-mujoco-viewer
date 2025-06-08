"""Checkbox panel for run-time visual settings."""

from typing import Callable

import mujoco
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QWidget,
)


class SettingsWidget(QWidget):
    """Manages the settings panel for the MuJoCo Viewer."""

    def __init__(
        self,
        *,
        get_set_flag: Callable[[int, bool], None],
        set_opt_attr: Callable[[str, int], None],
        force_init: bool,
        point_init: bool,
        inertia_init: bool,
        joint_init: bool,
        label_init: int,
        frame_init: int,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        self._set_flag = get_set_flag
        self._set_opt = set_opt_attr

        self._chk_force = QCheckBox("Contact Forces")
        self._chk_force.setChecked(force_init)
        self._chk_force.toggled.connect(lambda b: self._set_flag(mujoco.mjtVisFlag.mjVIS_CONTACTFORCE, b))

        self._chk_point = QCheckBox("Contact Points")
        self._chk_point.setChecked(point_init)
        self._chk_point.toggled.connect(lambda b: self._set_flag(mujoco.mjtVisFlag.mjVIS_CONTACTPOINT, b))

        self._chk_inertia = QCheckBox("Inertia Ellipsoids")
        self._chk_inertia.setChecked(inertia_init)
        self._chk_inertia.toggled.connect(lambda b: self._set_flag(mujoco.mjtVisFlag.mjVIS_INERTIA, b))

        self._chk_joint = QCheckBox("Joint Axes")
        self._chk_joint.setChecked(joint_init)
        self._chk_joint.toggled.connect(lambda b: self._set_flag(mujoco.mjtVisFlag.mjVIS_JOINT, b))

        lay = QFormLayout(self)
        lay.addRow(self._chk_force)
        lay.addRow(self._chk_point)
        lay.addRow(self._chk_inertia)
        lay.addRow(self._chk_joint)

        # Object-label drop-down
        lbl_box = QComboBox()
        lbl_box.addItem("None", mujoco.mjtLabel.mjLABEL_NONE)
        lbl_box.addItem("Body names", mujoco.mjtLabel.mjLABEL_BODY)
        lbl_box.addItem("Geom names", mujoco.mjtLabel.mjLABEL_GEOM)
        lbl_box.addItem("Site names", mujoco.mjtLabel.mjLABEL_SITE)
        idx = next(i for i in range(lbl_box.count()) if lbl_box.itemData(i) == label_init)
        lbl_box.setCurrentIndex(idx)
        lbl_box.currentIndexChanged.connect(lambda i: self._set_opt("label", lbl_box.itemData(i)))
        lay.addRow("Object labels:", lbl_box)

        # Spatial-frame drop-down
        frm_box = QComboBox()
        frm_box.addItem("None", mujoco.mjtFrame.mjFRAME_NONE)
        frm_box.addItem("World frame", mujoco.mjtFrame.mjFRAME_WORLD)
        frm_box.addItem("Body frames", mujoco.mjtFrame.mjFRAME_BODY)
        frm_box.addItem("Geom frames", mujoco.mjtFrame.mjFRAME_GEOM)
        frm_box.addItem("Site frames", mujoco.mjtFrame.mjFRAME_SITE)
        idx = next(i for i in range(frm_box.count()) if frm_box.itemData(i) == frame_init)
        frm_box.setCurrentIndex(idx)
        frm_box.currentIndexChanged.connect(lambda i: self._set_opt("frame", frm_box.itemData(i)))
        lay.addRow("Spatial frames:", frm_box)
