"""kmv – K-Scale MuJoCo Viewer (minimal skeleton)."""

__version__ = "0.2.1"

from .viewer import Viewer, launch          # re-export public API

__all__ = ["Viewer", "launch"]