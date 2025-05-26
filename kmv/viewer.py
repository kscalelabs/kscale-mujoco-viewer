"""Single-window convenience wrapper."""

from __future__ import annotations
import sys
import logging
from pathlib import Path

from PySide6.QtWidgets import QApplication, QMainWindow, QWidget
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
import mujoco

from .engine import SimEngine
from .renderer import GLViewport
from .plots import QPosPlot

logger = logging.getLogger(__name__)


class Viewer(QMainWindow):
    """Drag-and-drop MuJoCo XML viewer (with live qpos[0] plot)."""

    def __init__(self, xml_text: str, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("KMV viewer")

        try:
            # --- build MuJoCo world ------------------------------------------------
            logger.info("Creating MuJoCo model from XML...")
            self.model = mujoco.MjModel.from_xml_string(xml_text)
            logger.info("MuJoCo model created successfully")
            
            logger.info("Creating MuJoCo data...")
            self.data  = mujoco.MjData(self.model)
            logger.info("MuJoCo data created successfully")

            # --- sub-components ----------------------------------------------------
            logger.info("Creating SimEngine...")
            self.engine   = SimEngine(self.model, self.data, hz=1000, parent=self)
            logger.info("SimEngine created successfully")
            
            logger.info("Creating GLViewport...")
            self.viewport = GLViewport(self.model, self.data)
            logger.info("GLViewport created successfully")

            # When physics steps, ask viewport to repaint ASAP
            self.engine.stepped.connect(self.viewport.request_update)

            # 1. embed the GL viewport
            logger.info("Creating window container...")
            container = QWidget.createWindowContainer(self.viewport, self)
            self.setCentralWidget(container)
            logger.info("Window container created successfully")

            # 2. create plot dock and wire it to the physics clock
            logger.info("Creating plot dock...")
            self._plot_dock = QPosPlot(self.model, self.data, parent=self)
            self.addDockWidget(Qt.RightDockWidgetArea, self._plot_dock)
            self.engine.stepped.connect(self._plot_dock.on_step)
            logger.info("Plot dock created and connected successfully")

            # --- toolbar with Reset button --------------------------------------
            logger.info("Creating toolbar...")
            tb = self.addToolBar("Sim")
            reset_act = QAction("Reset", self)
            reset_act.setShortcut("R")
            reset_act.triggered.connect(self._reset_sim)
            tb.addAction(reset_act)
            logger.info("Toolbar created successfully")
            
            logger.info("Viewer initialization complete")

        except Exception as e:
            logger.error(f"Error during Viewer initialization: {e}")
            import traceback
            traceback.print_exc()
            raise

    # public helper ------------------------------------------------------------
    @staticmethod
    def from_path(path: str | Path) -> "Viewer":
        return Viewer(Path(path).read_text())

    # ------------------------------------------------------------------ #
    # private slot
    # ------------------------------------------------------------------ #
    def _reset_sim(self) -> None:
        """Reset physics + clear plots."""
        self.engine.reset()
        self._plot_dock.reset()


# --------------------------------------------------------------------------- #
# convenience launcher
# --------------------------------------------------------------------------- #
def launch(xml_text: str) -> None:      # retained for `import kmv; kmv.launch`
    # Set up logging if not already configured
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    try:
        logger.info("Creating QApplication...")
        app = QApplication.instance() or QApplication(sys.argv)
        logger.info("QApplication created")
        
        logger.info("Creating Viewer window...")
        win = Viewer(xml_text)
        logger.info("Viewer window created")
        
        logger.info("Resizing and showing window...")
        win.resize(800, 600)
        win.show()
        logger.info("Window shown, starting event loop...")
        
        sys.exit(app.exec())
    except Exception as e:
        logger.error(f"Error in launch function: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
