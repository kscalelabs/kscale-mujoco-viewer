"""Status bar widget for displaying simulation metrics and timing information."""

from __future__ import annotations
import time
from typing import TYPE_CHECKING

from PySide6.QtWidgets import QStatusBar

if TYPE_CHECKING:
    from kmv.core.buffer import RingBuffer
    from kmv.core.types import Frame


class SimulationStatusBar:
    """Manages status bar updates for the MuJoCo viewer."""
    
    def __init__(self, status_bar: QStatusBar) -> None:
        self._status_bar = status_bar
        self._fps_timer = time.time()
        self._frame_ctr = 0
        self._last_sim_time: float = 0.0
        self._last_absolute_sim_time: float = 0.0
        
    def update_fps_and_timing(
        self,
        ringbuffer: RingBuffer[Frame] | None = None,
        sim_time: float | None = None,
        absolute_sim_time: float | None = None,
    ) -> None:
        """Update the status bar with FPS and timing information."""
        self._frame_ctr += 1
        
        # Update timing values if provided
        if sim_time is not None:
            self._last_sim_time = sim_time
        if absolute_sim_time is not None:
            self._last_absolute_sim_time = absolute_sim_time
        
        # Update status bar every second
        if time.time() - self._fps_timer >= 1.0:
            self._update_status_message(ringbuffer)
            self._frame_ctr = 0
            self._fps_timer = time.time()
    
    def _update_status_message(self, ringbuffer: RingBuffer[Frame] | None = None) -> None:
        """Construct and display the status message."""
        # Base FPS information
        fps_msg = f"{self._frame_ctr} FPS"
        
        # Add ringbuffer statistics if available
        if ringbuffer is not None:
            pushes = ringbuffer.push_count
            pops = ringbuffer.pop_count
            dropped = pushes - pops
            backlog = len(ringbuffer)
            fps_msg += f"   P:{pushes}  C:{pops}  Δ:{dropped}  len:{backlog}"
        
        # Add timing information
        timing_msg = f"   sim_t:{self._last_sim_time:.3f}   abs_t:{self._last_absolute_sim_time:.3f}"
        
        # Combine all information
        full_message = fps_msg + timing_msg
        self._status_bar.showMessage(full_message)
    
    def set_sim_time(self, sim_time: float) -> None:
        """Update the current simulation time."""
        self._last_sim_time = sim_time
    
    def set_absolute_sim_time(self, absolute_sim_time: float) -> None:
        """Update the current absolute simulation time."""
        self._last_absolute_sim_time = absolute_sim_time
    
    def show_message(self, message: str, timeout: int = 0) -> None:
        """Show a custom message in the status bar."""
        self._status_bar.showMessage(message, timeout)
