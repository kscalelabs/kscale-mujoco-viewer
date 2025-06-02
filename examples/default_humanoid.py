#!/usr/bin/env python3
"""
Sanity-check for the new *out-of-process* viewer.

Runs the MuJoCo humanoid for a while and streams qpos/qvel to
`RemoteViewer`.  The GUI should stay ~60 FPS even if we sleep.
"""

import time
from pathlib import Path
import mujoco
import numpy as np
import colorlogging
import logging

from kmv.app.remote import RemoteViewer          # ← NEW viewer handle

logger = logging.getLogger(__name__)


def main() -> None:
    colorlogging.configure()
    xml_path = Path(__file__).parent.parent / "tests" / "assets" / "humanoid.xml"
    logger.info("Loading model: %s", xml_path)

    physics_dt   = 0.002
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    model.opt.timestep = physics_dt
    data  = mujoco.MjData(model)

    viewer = RemoteViewer(model, xml_path) # starts its own process

    # camera defaults (same as before)
    viewer.push_frame(qpos=data.qpos.copy(), qvel=data.qvel.copy())  # send one frame so GUI opens
    logger.info("Viewer launched, hit Ctrl-C to quit")

    counter = 0
    frames_sent = 0
    wall_t0 = time.time()

    try:
        while True:

            counter += 1

            mujoco.mj_step(model, data)
            
            # Push the frame to the viewer
            viewer.push_frame(
                qpos=data.qpos.copy(),
                qvel=data.qvel.copy(),
                sim_time=float(data.time),
            )
            frames_sent += 1
            
            wall_time_elapsed = time.time() - wall_t0
            real_time_factor = float(data.time) / (wall_time_elapsed + 1e-9)
            
            viewer.push_metrics(
                {
                    "sim t":            float(data.time),
                    "real t":           wall_time_elapsed,
                    "iters":            counter,
                    "frames sent":      frames_sent,
                    "× real-time":      real_time_factor,
                }
            )
            
            time.sleep(0.000001)

    except KeyboardInterrupt:
        print("\nExiting.")

    finally:
        viewer.close()


if __name__ == "__main__":
    main()
