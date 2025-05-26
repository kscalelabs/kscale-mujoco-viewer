"""Physics clock kept separate from all OpenGL work."""

from PySide6.QtCore import QObject, QTimer, Signal
import mujoco


class SimEngine(QObject):
    """Steps MuJoCo at a fixed *wall-clock* rate (default 1 kHz)."""

    stepped = Signal(float)        # emits simulation time after each mj_step

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData | None = None,
                 hz: int = 1000, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.model = model
        self.data = data or mujoco.MjData(model)

        self._timer = QTimer(self, timeout=self._step)
        self.set_rate(hz)

    # --------------------------------------------------------------------- #
    # public helpers
    # --------------------------------------------------------------------- #
    def set_rate(self, hz: int) -> None:
        """Change stepping frequency on the fly (Hz of *sim-time*)."""
        self._dt_ms = int(1000 / hz)
        self._timer.start(self._dt_ms)

    # --------------------------------------------------------------------- #
    # private
    # --------------------------------------------------------------------- #
    def _step(self) -> None:
        mujoco.mj_step(self.model, self.data)
        self.stepped.emit(self.data.time)
        
    def reset(self) -> None:
        """Return the model to its XML initial state and emit a fresh step."""
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)         # recompute derived fields
        self.stepped.emit(self.data.time)                # refresh all listeners