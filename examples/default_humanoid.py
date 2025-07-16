"""Example script that spawns a humanoid model in the viewer.

Runs MuJoCo's default humanoid for a while and streams qpos/qvel + live telemetry
to `kmv.app.viewer.Viewer`.  The GUI should stay ~60 FPS even if we sleep.
"""

import logging
import time
from pathlib import Path

import colorlogging
import mujoco
import numpy as np
from collections import deque

from kmv.app.viewer import QtViewer
from kmv.core.types import GeomType, Marker

logger = logging.getLogger(__name__)

PHYSICS_DT = 0.02
TRAIL_LEN = 150          # max # of capsule segments to keep
STEP_SKIP = 3            # decimate updates (every N sim steps)


def run_default_humanoid() -> None:
    """Run the default humanoid simulation."""
    xml_path = Path(__file__).parent.parent / "tests" / "assets" / "humanoid.xml"
    logger.info("Loading model: %s", xml_path)
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    model.opt.timestep = PHYSICS_DT
    data = mujoco.MjData(model)

    viewer = QtViewer(model)

    # ------------------------------------------------------------------
    #  Trail bookkeeping
    # ------------------------------------------------------------------
    trail_pts: deque[np.ndarray] = deque(maxlen=TRAIL_LEN + 1)  # vertices
    trail_ids: deque[str] = deque(maxlen=TRAIL_LEN)             # marker IDs
    next_seg = 0

    def _make_segment(p0: np.ndarray, p1: np.ndarray, seg_id: str) -> Marker:
        """Create a world-space capsule whose axis runs from p0 → p1."""
        mid = (p0 + p1) * 0.5
        d   = p1 - p0
        length = float(np.linalg.norm(d))
        if length < 1e-9:                # guard for zero-length
            return Marker(id=seg_id, pos=tuple(mid))

        # Build a rotation matrix that aligns +z with the segment direction
        z = d / length
        x = np.cross([0, 0, 1], z)
        if np.linalg.norm(x) < 1e-9:     # collinear with world-z
            x = np.array([1, 0, 0])
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        rot = np.stack([x, y, z], axis=1).reshape(-1)   # row-major 9-tuple

        return Marker(
            id=seg_id,
            pos=tuple(mid),
            geom_type=GeomType.CAPSULE,
            size=(0.01, length * 0.5, 0.01),     # (radius, half-len, radius)
            rgba=(0.1, 0.6, 1.0, 0.9),           # light-blue, slight alpha
            orient=tuple(rot),                   # ← **key change**
        )

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
    viewer.add_marker(
        Marker(
            id="torso_arrow",
            body_id=body_id,
            local_offset=(0, 0, 0.2),
            geom_type=GeomType.ARROW,
            size=(0.02, 0.20, 0.02),
            rgba=(0, 1, 0, 1),
        )
    )

    viewer.add_marker(
        Marker(id="red_sphere", pos=(0, 0, 0), geom_type=GeomType.SPHERE, size=(0.05, 0.05, 0.05), rgba=(1, 0, 0, 1))
    )

    logger.info("Viewer launched — Ctrl-drag to perturb, hit Ctrl-C or close window to quit.")

    sim_it_counter = 0
    t0_wall = time.perf_counter()

    try:
        while True:
            sim_it_counter += 1

            # Physics step
            mujoco.mj_step(model, data)

            # Stream state to viewer
            viewer.push_state(data.qpos, data.qvel, sim_time=data.time)

            # Apply push forces from the viewer
            xfrc = viewer.drain_control_pipe()
            if xfrc is not None:
                data.xfrc_applied[:] = xfrc

            # Plot stuff
            viewer.push_plot_metrics(
                group="Physics",
                scalars={
                    "qpos0": float(data.qpos[0]),
                    "qpos1": float(data.qpos[1]),
                    "qpos2": float(data.qpos[2]),
                    "qvel0": float(data.qvel[0]),
                    "qvel1": float(data.qvel[1]),
                    "qvel2": float(data.qvel[2]),
                },
            )

            # ----------------------------------------------------------
            #  Live torso trail (capsule segments)
            # ----------------------------------------------------------
            torso_xyz = data.xpos[body_id].copy()
            if sim_it_counter % STEP_SKIP == 0:
                trail_pts.append(torso_xyz)

                if len(trail_pts) >= 2:
                    p0, p1 = trail_pts[-2], trail_pts[-1]
                    seg_id = f"trail_{next_seg}"
                    next_seg += 1
                    viewer.add_marker(_make_segment(p0, p1, seg_id))
                    trail_ids.append(seg_id)

                # Keep the list bounded
                if len(trail_ids) == TRAIL_LEN:
                    viewer.remove_marker(trail_ids.popleft())

            if sim_it_counter % 100 == 0:
                viewer.update_marker("torso_arrow", rgba=(1, 0, 0, 1))

            # Sleep so that sim-time == wall-time
            target_wall = t0_wall + (data.time + PHYSICS_DT)
            now = time.perf_counter()
            sleep_time_seconds = target_wall - now
            if sleep_time_seconds > 0.0:
                time.sleep(sleep_time_seconds)

            if not viewer.is_open:
                logger.info("Viewer closed, exiting simulation loop.")
                break

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt, exiting simulation loop.")

    finally:
        viewer.close()


def main() -> None:
    colorlogging.configure()
    run_default_humanoid()


if __name__ == "__main__":
    main()
