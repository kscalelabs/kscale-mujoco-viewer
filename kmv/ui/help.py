"""Help widget for displaying viewer instructions."""

from PySide6.QtWidgets import QTextEdit, QVBoxLayout, QWidget


class HelpWidget(QWidget):
    """Widget that displays help instructions for the MuJoCo Viewer."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        layout = QVBoxLayout(self)

        # Create text area for help content
        self._text_edit = QTextEdit(self)
        self._text_edit.setReadOnly(True)
        self._text_edit.setHtml(self._get_help_content())

        layout.addWidget(self._text_edit)

    def _get_help_content(self) -> str:
        """Returns the HTML-formatted help content."""
        return """
        <style>
            li {
                margin-bottom: 8px;
            }
            ul {
                margin-bottom: 26px;
            }
            h3 {
                margin-top: 20px;
                margin-bottom: 12px;
            }
        </style>

        <h2>K-Scale MuJoCo Viewer - Help</h2>

        <h3>📷 Camera Controls</h3>
        <ul>
            <li><b>Mouse Left Drag:</b> Rotate camera</li>
            <li><b>Mouse Right Drag:</b> Pan camera</li>
            <li><b>Mouse Scroll:</b> Zoom in/out</li>
        </ul>

        <h3>🥊 Push Controls </h3>
        <ul>
            <li><b>Ctrl + Mouse Left Drag:</b> Apply a rotational force to a body</li>
            <li><b>Ctrl + Mouse Right Drag:</b> Apply a linear force to a body vertically</li>
            <li><b>Ctrl + Shift + Right Drag:</b> Apply a linear force to a body horizontally</li>
        </ul>

        <h3>⚙️ Menus</h3>
        <ul>
            <li><b>Plots:</b> Toggle different plot groups on/off</li>
            <li><b>Viewer Stats:</b> Show/hide viewer statistics table</li>
            <li><b>Help:</b> Display this help information</li>
            <li><b>Settings:</b> Show/hide real-time visual settings</li>
        </ul>

        <h3>🔧 Settings Panel</h3>
        <p>The Settings dock lets you toggle visual overlays and controls on the fly:</p>
        <ul>
            <li><b>Contact Forces:</b> Draw coloured arrows indicating contact
                impulses between geoms.</li>
            <li><b>Contact Points:</b> Mark each active contact with a small dot.</li>
            <li><b>Inertia Ellipsoids:</b> Visualize the inertial properties of bodies
                as 3D ellipsoids.</li>
            <li><b>Joint Axes:</b> Display coordinate axes at joint locations
                showing joint orientation.</li>
            <li><b>Object labels:</b> Choose which object names to display - None,
                Body names, Geom names, or Site names.</li>
            <li><b>Spatial frames:</b> Select which coordinate frames to show - None,
                World frame, Body frames, Geom frames, or Site frames.</li>
        </ul>

        <h3>📈 Viewer Stats</h3>
        <p>The viewer stats table provides detailed performance metrics:</p>
        <ul>
            <li><b>Viewer FPS:</b> Frames per second rendered by the viewer</li>
            <li><b>Plot FPS:</b> Rate at which plot data is updated per second</li>
            <li><b>Phys Iters/s:</b> Physics simulation iterations per second from calling process</li>
            <li><b>Abs Sim Time:</b> Total absolute simulation time including resets</li>
            <li><b>Sim Time / Real Time:</b> Simulation speed ratio compared to real-time</li>
            <li><b>Wall Time:</b> Actual elapsed wall-clock time since viewer started</li>
            <li><b>Reset Count:</b> Total number of simulation environment resets</li>
        </ul>

        <p><i>For more information, visit the K-Scale documentation.</i></p>
        """
