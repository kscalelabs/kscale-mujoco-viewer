"""Single-window convenience wrapper."""

from __future__ import annotations
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication, QMainWindow, QWidget
import mujoco

from .engine import SimEngine
from .renderer import GLViewport


class Viewer(QMainWindow):
    """Drag-and-drop MuJoCo XML viewer (minimal)."""

    def __init__(self, xml_text: str, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("KMV viewer")

        try:
            # --- build MuJoCo world ------------------------------------------------
            print("Creating MuJoCo model from XML...")
            self.model = mujoco.MjModel.from_xml_string(xml_text)
            print("MuJoCo model created successfully")
            
            print("Creating MuJoCo data...")
            self.data  = mujoco.MjData(self.model)
            print("MuJoCo data created successfully")

            # --- sub-components ----------------------------------------------------
            print("Creating SimEngine...")
            self.engine   = SimEngine(self.model, self.data, hz=1000, parent=self)
            print("SimEngine created successfully")
            
            print("Creating GLViewport...")
            self.viewport = GLViewport(self.model, self.data)
            print("GLViewport created successfully")

            # When physics steps, ask viewport to repaint ASAP
            self.engine.stepped.connect(self.viewport.request_update)

            # Embed the QOpenGLWindow in the main window using createWindowContainer
            print("Creating window container...")
            container = QWidget.createWindowContainer(self.viewport, self)
            self.setCentralWidget(container)
            print("Viewer initialization complete")

        except Exception as e:
            print(f"Error during Viewer initialization: {e}")
            import traceback
            traceback.print_exc()
            raise

    # public helper ------------------------------------------------------------
    @staticmethod
    def from_path(path: str | Path) -> "Viewer":
        return Viewer(Path(path).read_text())


# --------------------------------------------------------------------------- #
# convenience launcher
# --------------------------------------------------------------------------- #
def launch(xml_text: str) -> None:      # retained for `import kmv; kmv.launch`
    try:
        print("Creating QApplication...")
        app = QApplication.instance() or QApplication(sys.argv)
        print("QApplication created")
        
        print("Creating Viewer window...")
        win = Viewer(xml_text)
        print("Viewer window created")
        
        print("Resizing and showing window...")
        win.resize(800, 600)
        win.show()
        print("Window shown, starting event loop...")
        
        sys.exit(app.exec())
    except Exception as e:
        print(f"Error in launch function: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
