"""Draw various markers in the viewer."""

from dataclasses import dataclass
from typing import Tuple

RGBA = Tuple[float, float, float, float]


@dataclass(slots=True)
class SphereMarker:
    pos: Tuple[float, float, float]
    radius: float = 0.05
    rgba: RGBA = (1.0, 0.0, 0.0, 1.0)
