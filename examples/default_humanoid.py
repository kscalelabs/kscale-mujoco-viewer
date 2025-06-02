#!/usr/bin/env python3
"""
Sanity-check for the new *out-of-process* viewer.

Runs MuJoCo’s humanoid for a while and streams qpos/qvel + live telemetry
to `kmv.app.viewer.Viewer`.  The GUI should stay ~60 FPS even if we sleep.
"""

from __future__ import annotations

import time
from pathlib import Path
import logging

import mujoco
import numpy as np
import colorlogging

from kmv.app.viewer import Viewer

LOGGER = logging.getLogger(__name__)


def main() -> None:
    # ---------- logging ------------------------------------------------ #
    colorlogging.configure()
    xml_path = Path(__file__).parent.parent / "tests" / "assets" / "humanoid.xml"
    LOGGER.info("Loading model: %s", xml_path)

    # ---------- compile model ----------------------------------------- #
    physics_dt = 0.002
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    model.opt.timestep = physics_dt
    data  = mujoco.MjData(model)

    # ---------- launch viewer ----------------------------------------- #
    viewer = Viewer(model, enable_plots=True)

    # push one frame so GUI window pops immediately
    viewer.push_state(data.qpos, data.qvel, sim_time=data.time)

    LOGGER.info("Viewer launched — Ctrl-drag to perturb, hit Ctrl-C to quit")

    # ---------- main loop --------------------------------------------- #
    sim_it  = 0
    t0_wall = time.time()
    pending_xfrc = None

    try:
        while True:
            sim_it += 1

            # (1) apply any force returned from GUI -------------------- #
            if pending_xfrc is not None:
                data.xfrc_applied[:] = pending_xfrc
                pending_xfrc = None                 # MuJoCo will clear after step

            # (2) step physics ---------------------------------------- #
            mujoco.mj_step(model, data)

            # (3) send newest state to viewer ------------------------- #
            viewer.push_state(data.qpos, data.qvel, sim_time=data.time)

            # (4) pull GUI-generated forces --------------------------- #
            new_force = viewer.poll_forces()
            if new_force is not None:
                pending_xfrc = new_force

            # (5) live telemetry -------------------------------------- #
            wall_elapsed = time.time() - t0_wall
            realtime_x   = float(data.time) / (wall_elapsed + 1e-9)
            viewer.push_scalars(
                {
                    "sim t":        float(data.time),
                    "wall t":       wall_elapsed,
                    "iters":        sim_it,
                    "× real-time":  realtime_x,
                }
            )

            time.sleep(0.001)   # ~1 kHz sim → render throttled to 60 Hz

    except KeyboardInterrupt:
        print("\nExiting simulation loop…")

    finally:
        viewer.close()


if __name__ == "__main__":
    main()
