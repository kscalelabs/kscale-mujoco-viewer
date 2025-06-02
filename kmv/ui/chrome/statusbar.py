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
        # ── FPS bookkeeping ────────────────────────────────────────────
        self._fps_timer = time.time()
        self._frame_ctr = 0
        self._last_fps = 0  # Store the last calculated FPS

        # ── make the whole bar monospaced so widths don't vary ──────────
        # Works on all platforms that ship 'Courier New' or fall back to
        # the default monospace.  Feel free to replace with your favourite.
        status_bar.setStyleSheet(
            "QStatusBar { font-family: 'Courier New', monospace; }"
        )
        # ── real-time reference – first paint = t₀ ─────────────────────
        self._t0_real = self._fps_timer
        
    def update_fps_and_timing(
        self,
        ringbuffer: RingBuffer[Frame] | None = None,
        sim_time: float = 0.0,
        absolute_sim_time: float = 0.0,
    ) -> None:
        """Update the status bar with FPS and timing information."""
        self._frame_ctr += 1
        
        # Calculate FPS every second
        current_time = time.time()
        if current_time - self._fps_timer >= 1.0:
            self._last_fps = self._frame_ctr
            self._frame_ctr = 0
            self._fps_timer = current_time
        
        # Update timing display every iteration (real-time)
        self._update_status_message(ringbuffer, sim_time, absolute_sim_time)
    
    def _update_status_message(
        self, 
        ringbuffer: RingBuffer[Frame] | None = None,
        sim_time: float = 0.0,
        absolute_sim_time: float = 0.0,
    ) -> None:
        """Construct and display the status message."""
        # Base FPS information (fixed width: 4-digit int)
        fps_msg = f"{self._last_fps:4d} FPS"
        
        # Add ringbuffer statistics if available
        if ringbuffer is not None:
            pushes = ringbuffer.push_count
            pops = ringbuffer.pop_count
            dropped = pushes - pops
            backlog = len(ringbuffer)
            fps_msg += (
                f"   P:{pushes:6d}"
                f"  C:{pops:6d}"
                f"  Δ:{dropped:6d}"
                f"  len:{backlog:3d}"
            )
        
        # Add timing information (updates every iteration)
        real_elapsed = time.time() - self._t0_real
        timing_msg = (
            f"   sim_t:{sim_time:>9.3f}"
            f"   abs_t:{absolute_sim_time:>9.3f}"
            f"   real_t:{real_elapsed:>9.3f}"
        )
        
        # Combine all information
        full_message = fps_msg + timing_msg
        self._status_bar.showMessage(full_message)
    
    def show_message(self, message: str, timeout: int = 0) -> None:
        """Show a custom message in the status bar."""
        self._status_bar.showMessage(message, timeout)
