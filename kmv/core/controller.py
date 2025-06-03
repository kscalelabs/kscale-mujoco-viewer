from __future__ import annotations
import time
from typing import Mapping, Callable

import numpy as np
import mujoco

from kmv.ipc.state import SharedArrayRing
from kmv.core.types import Msg, ForcePacket, TelemetryPacket, PlotPacket

_Array = np.ndarray
_Scalars = Mapping[str, float]
_OnForces = Callable[[_Array], None]


class RenderLoop:
    """Pure-Python, no-Qt controller that owns the render-time state machine."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        rings: Mapping[str, SharedArrayRing],
        *,
        on_forces: _OnForces | None = None,
        get_table: Callable[[], TelemetryPacket | None],
        get_plot:  Callable[[], PlotPacket | None],
    ) -> None:
        self._model, self._data = model, data
        self._rings = rings
        self._on_forces = on_forces
        self._get_table = get_table
        self._get_plot  = get_plot

        # ── perf & bookkeeping ──────────────────────────────
        self._fps_timer   = time.perf_counter()
        self._frame_ctr   = 0
        self.fps          = 0.0

        self._plot_timer  = time.perf_counter()
        self._plot_ctr    = 0
        self.plot_hz      = 0.0

        self._phys_iters_prev      = 0
        self._phys_iters_prev_time = time.perf_counter()
        self.phys_iters_per_sec    = 0.0

        self._wall_start : float | None = None

        self.sim_time_abs = 0.0
        self.reset_count  = 0
        self._sim_prev    = 0.0
        self._sim_offset  = 0.0
        self._reset_tol   = 1e-9

        # latest GUI consumables
        self._last_table: dict[str, float] = {}
        self._plots_latest: dict[str, _Scalars] = {}
        self._last_plot: PlotPacket | None = None   # ← kept for legacy access

    # ------------------------------------------------------------------ #
    # public façade – one call per GUI frame
    # ------------------------------------------------------------------ #
    def tick(self) -> None:
        """Advance state **and** update `mjData` in-place."""

        # 1.  Pull latest physics
        self._pull_state()

        # 2.  Drain metrics queues
        self._drain_metrics()

        # 3.  Recompute derived counters (FPS etc.)
        self._account_timing()

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    def _pull_state(self) -> None:
        qpos = self._rings["qpos"].latest()
        qvel = self._rings["qvel"].latest()
        sim_time = float(self._rings["sim_time"].latest()[0])

        if sim_time < self._sim_prev - self._reset_tol:
            self._sim_offset += self._sim_prev
            self.reset_count += 1
        self._sim_prev    = sim_time
        self.sim_time_abs = self._sim_offset + sim_time

        self._data.qpos[:]  = qpos
        self._data.qvel[:]  = qvel
        self._data.time     = sim_time
        mujoco.mj_forward(self._model, self._data)

    def _drain_metrics(self) -> None:
        while (pkt := self._get_table()) is not None:
            self._last_table.update(pkt.rows)

            # live physics-throughput
            if "Phys Iters" in pkt.rows:
                now = time.perf_counter()
                dt  = now - self._phys_iters_prev_time
                if dt > 0:
                    self.phys_iters_per_sec = (
                        pkt.rows["Phys Iters"] - self._phys_iters_prev
                    ) / dt
                self._phys_iters_prev      = pkt.rows["Phys Iters"]
                self._phys_iters_prev_time = now

        while (pkt := self._get_plot()) is not None:
            self._plots_latest[pkt.group] = pkt.scalars   # remember per-group
            self._last_plot = pkt                        # backward-compat
            self._plot_ctr += 1

    def _account_timing(self) -> None:
        self._frame_ctr += 1
        now = time.perf_counter()
        if (now - self._fps_timer) >= 1.0:
            self.fps = self._frame_ctr / (now - self._fps_timer)
            self._frame_ctr = 0
            self._fps_timer = now

        if (now - self._plot_timer) >= 1.0:
            self.plot_hz   = self._plot_ctr / (now - self._plot_timer)
            self._plot_ctr = 0
            self._plot_timer = now

        # wall-clock bookkeeping
        if self._wall_start is None:
            self._wall_start = now
        wall_elapsed = now - self._wall_start
        realtime_x   = (
            self.sim_time_abs / max(wall_elapsed, 1e-9)
            if wall_elapsed > 0 else 0.0
        )

        # keep a ready-to-use row dict
        self._last_table.update(
            {
                "Viewer FPS":          round(self.fps,       1),
                "Plot FPS":            round(self.plot_hz,   1),
                "Phys Iters/s":        round(self.phys_iters_per_sec, 1),
                "Abs Sim Time":        round(self.sim_time_abs, 3),
                "Sim Time / Real Time":round(realtime_x, 2),
                "Wall Time":           round(wall_elapsed, 2),
                "Reset Count":         self.reset_count,
            }
        )
