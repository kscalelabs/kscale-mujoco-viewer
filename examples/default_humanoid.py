"""Example script that spawns a humanoid model in the viewer.

Runs MuJoCo's default humanoid for a while and streams qpos/qvel + live telemetry
to `kmv.app.viewer.Viewer`.  The GUI should stay ~60 FPS even if we sleep.
"""

import logging
import time
from pathlib import Path

import colorlogging
import mujoco

from kmv.app.viewer import QtViewer
from kmv.core.types import GeomType, Marker

logger = logging.getLogger(__name__)

PHYSICS_DT = 0.02


def run_default_humanoid() -> None:
    """Run the default humanoid simulation."""
    xml_path = Path(__file__).parent.parent / "tests" / "assets" / "humanoid.xml"
    logger.info("Loading model: %s", xml_path)
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    model.opt.timestep = PHYSICS_DT
    data = mujoco.MjData(model)

    viewer = QtViewer(model)
    logger.info("Viewer launched — Ctrl-drag to perturb, hit Ctrl-C or close window to quit.")

    # How to add a marker that automatically tracks the torso
    torso_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
    viewer.add_marker(
        Marker(
            id="torso_arrow",
            body_id=torso_body_id,
            local_offset=(0, 0, 0.35),
            geom_type=GeomType.ARROW,
            size=(0.02, 0.020, 0.2),
            rgba=(0, 1, 0, 1),
        )
    )

    # How to add a marker that just stays put
    viewer.add_marker(
        Marker(id="red_sphere", pos=(0, 0, 0), geom_type=GeomType.SPHERE, size=(0.05, 0.05, 0.05), rgba=(1, 0, 0, 1))
    )

    # How to add a trail that automatically tracks the torso
    viewer.add_trail(
        "torso_path",
        track_body_id=torso_body_id,
        max_len=300,
        radius=0.01,
    )

    # How to add a trail where you can manually push points
    # Note how im not passing the track_body_id argument
    # We will manually push points to this trail in the simulation loop
    left_hand_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand_left")
    viewer.add_trail("left_hand_path_manual", max_len=100, radius=0.01, min_segment_dist=0.01, rgba=(1, 0, 1, 1))

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

            left_hand_xyz = data.xpos[left_hand_body_id].copy()
            viewer.push_trail_point("left_hand_path_manual", tuple(left_hand_xyz))

            # You can also update markers in place
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
