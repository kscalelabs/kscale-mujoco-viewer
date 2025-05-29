"""kmv – K-Scale MuJoCo Viewer (minimal skeleton)."""

__version__ = "0.2.1"

from .qtviewer import QtViewer, launch_interactive_viewer  # Main ksim-compatible viewer

__all__ = ["QtViewer", "launch_interactive_viewer"]