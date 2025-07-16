"""Render-side runtime utilities for the KMV GUI worker.

The module runs entirely inside the **viewer process**: it reads physics
streams from shared-memory rings, drains metric queues, and keeps a rolling
set of timing / telemetry values that the Qt widgets display.
"""

import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Mapping

import mujoco
import numpy as np

from kmv.core.types import (
    AddTrail,
    Marker,
    PlotPacket,
    PushTrailPoint,
    RemoveTrail,
    TelemetryPacket,
    _MarkerCmd,
)
from kmv.ipc.shared_ring import SharedMemoryRing
from kmv.utils.geometry import capsule_between

_Array = np.ndarray
_Scalars = Mapping[str, float]
_OnForces = Callable[[_Array], None]


@dataclass
class _TrailState:
    max_len: int | None
    radius: float
    rgba: tuple[float, float, float, float]
    pts: deque[np.ndarray]  # all vertices (max_len + 1)
    seg_ids: deque[str]  # IDs of capsule markers
    next_seg: int = 0  # running counter


class RenderLoop:
    """Per-frame state manager for the GUI process.

    Each `tick()` pulls the newest *qpos/qvel* from shared memory, updates
    `mujoco.MjData`, ingests table/plot packets, and recomputes FPS and other
    diagnostics—all without touching Qt.  The viewport and widgets then read
    the freshly updated attributes to paint the current frame.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        rings: Mapping[str, SharedMemoryRing],
        *,
        on_forces: _OnForces | None = None,
        get_table: Callable[[], TelemetryPacket | None],
        get_plot: Callable[[], PlotPacket | None],
        get_markers: Callable[[], tuple[object, ...] | None],
    ) -> None:
        self._model, self._data = model, data
        self._rings = rings
        self._on_forces = on_forces
        self._get_table = get_table
        self._get_plot = get_plot
        self._get_markers = get_markers

        self._fps_timer = time.perf_counter()
        self._frame_ctr = 0
        self.fps = 0.0

        self._plot_timer = time.perf_counter()
        self._plot_ctr = 0
        self.plot_hz = 0.0

        self._phys_iters_prev: float = 0.0
        self._phys_iters_prev_time = time.perf_counter()
        self.phys_iters_per_sec = 0.0

        self._wall_start: float | None = None

        self.sim_time_abs = 0.0
        self.reset_count = 0
        self._sim_prev = 0.0
        self._sim_offset = 0.0
        self._reset_tol = 1e-9

        self._last_table: dict[str, float] = {}
        self._plots_latest: dict[str, _Scalars] = {}

        self._markers: dict[str | int, Marker] = {}
        self._trails: dict[str | int, _TrailState] = {}

    def tick(self) -> None:
        """Advance state and update `mjData` in-place."""
        self._pull_state()
        self._drain_metrics()
        self._drain_markers()
        self._account_timing()

    def _pull_state(self) -> None:
        """Pull the physics state from the parent process."""
        qpos = self._rings["qpos"].latest()
        qvel = self._rings["qvel"].latest()
        sim_time = float(self._rings["sim_time"].latest()[0])

        if sim_time < self._sim_prev - self._reset_tol:
            self._sim_offset += self._sim_prev
            self.reset_count += 1
        self._sim_prev = sim_time
        self.sim_time_abs = self._sim_offset + sim_time

        self._data.qpos[:] = qpos
        self._data.qvel[:] = qvel
        self._data.time = sim_time
        mujoco.mj_forward(self._model, self._data)

    def _drain_metrics(self) -> None:
        """Drain metrics from the parent process for the telemetry table."""
        while (pkt_table := self._get_table()) is not None:
            pkt = pkt_table  # keeps the original variable name but distinct type per loop
            self._last_table.update(pkt.rows)

            # Compute physics iterations per second
            if "Phys Iters" in pkt.rows:
                now = time.perf_counter()
                dt = now - self._phys_iters_prev_time
                if dt > 0:
                    self.phys_iters_per_sec = (pkt.rows["Phys Iters"] - self._phys_iters_prev) / dt
                self._phys_iters_prev = pkt.rows["Phys Iters"]
                self._phys_iters_prev_time = now

        # Drain plots
        while (pkt_plot := self._get_plot()) is not None:
            self._plots_latest[pkt_plot.group] = pkt_plot.scalars
            self._plot_ctr += 1

    def _drain_markers(self) -> None:
        """Handle marker *and* trail commands arriving over the queue."""
        while (cmd := self._get_markers()) is not None:
            # ----- ordinary markers -------------------------------------------------
            if isinstance(cmd, _MarkerCmd):
                cmd.apply(self._markers)
                continue

            # ----- trails -----------------------------------------------------------
            # 1) new trail
            if isinstance(cmd, AddTrail):
                self._trails[cmd.id] = _TrailState(
                    max_len=cmd.max_len,
                    radius=cmd.radius,
                    rgba=cmd.rgba,
                    pts=deque(maxlen=(cmd.max_len or 0) + 1),
                    seg_ids=deque(maxlen=(cmd.max_len or 0)),
                )
                continue

            # 2) push point
            if isinstance(cmd, PushTrailPoint):
                st = self._trails.get(cmd.id)
                if st is None:
                    continue  # silently ignore unknown trail
                pt = np.asarray(cmd.point, dtype=np.float64)
                st.pts.append(pt)

                # need ≥2 points to make a segment
                if len(st.pts) >= 2:
                    p0, p1 = st.pts[-2], st.pts[-1]
                    seg_id = f"{cmd.id}_{st.next_seg}"
                    st.next_seg += 1

                    self._markers[seg_id] = capsule_between(p0, p1, radius=st.radius, seg_id=seg_id, rgba=st.rgba)

                    # cap trail length **before** we lose the ID
                    if st.max_len is not None and len(st.seg_ids) >= st.max_len:
                        old_seg = st.seg_ids.popleft()  # now we still have it
                        self._markers.pop(old_seg, None)  # remove its Marker

                    st.seg_ids.append(seg_id)
                continue

            # 3) remove trail
            if isinstance(cmd, RemoveTrail):
                st = self._trails.pop(cmd.id, None)
                if st:
                    for seg_id in st.seg_ids:
                        self._markers.pop(seg_id, None)
                continue

    def _account_timing(self) -> None:
        """Account for timing metrics."""
        self._frame_ctr += 1
        now = time.perf_counter()

        # Viewer FPS
        if (now - self._fps_timer) >= 1.0:
            self.fps = self._frame_ctr / (now - self._fps_timer)
            self._frame_ctr = 0
            self._fps_timer = now

        # Plot FPS
        if (now - self._plot_timer) >= 1.0:
            self.plot_hz = self._plot_ctr / (now - self._plot_timer)
            self._plot_ctr = 0
            self._plot_timer = now

        # Wall-Time
        if self._wall_start is None:
            self._wall_start = now
        wall_elapsed = now - self._wall_start
        realtime_x = self.sim_time_abs / max(wall_elapsed, 1e-9) if wall_elapsed > 0 else 0.0

        self._last_table.update(
            {
                "Viewer FPS": round(self.fps, 1),
                "Plot FPS": round(self.plot_hz, 1),
                "Phys Iters/s": round(self.phys_iters_per_sec, 1),
                "Abs Sim Time": round(self.sim_time_abs, 3),
                "Sim Time / Real Time": round(realtime_x, 2),
                "Wall Time": round(wall_elapsed, 2),
                "Reset Count": self.reset_count,
            }
        )
