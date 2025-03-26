"""Example script demonstrating how to use the kscale-mujoco-viewer package with a humanoid model.

This example loads a humanoid model, sets up a viewer with camera tracking and real-time
plotting, and runs a simulation with random actions. It includes keyboard controls for
pausing/resuming the simulation and exiting.
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import colorlogging
import mujoco
import numpy as np
from pynput import keyboard

from kmv.viewer import MujocoViewerHandler, launch_passive

logger = logging.getLogger(__name__)

# Constants
DEFAULT_DURATION = 100.0  # seconds
DEFAULT_RENDER_WIDTH = 1280
DEFAULT_RENDER_HEIGHT = 720
CONTROL_DT = 0.02  # Control timestep (20ms)


class SimulationController:
    """Controls the simulation state through keyboard inputs.

    This class handles keyboard events for pausing/resuming simulation and exiting.
    """

    def __init__(self) -> None:
        """Initialize the simulation controller."""
        self.paused = False
        self.listener: keyboard.Listener | None = None

    def on_press(self, key: keyboard.Key | keyboard.KeyCode | None) -> None:
        """Handle keyboard press events.

        Args:
            key: The key that was pressed
        """
        if key == keyboard.Key.space:
            self.paused = not self.paused
            logger.info("Paused" if self.paused else "Resumed")
        elif key == keyboard.Key.esc:
            # Stop listener and exit
            if self.listener:
                self.listener.stop()
            logger.info("Exit signal received, closing simulation...")
            os._exit(0)  # Force exit the program

    def start_listener(self) -> keyboard.Listener:
        """Start the keyboard listener.

        Returns:
            The keyboard listener object
        """
        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()
        return self.listener

    def stop_listener(self) -> None:
        """Stop the keyboard listener safely."""
        if self.listener and self.listener.is_alive():
            self.listener.stop()


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="KScale MuJoCo Viewer Example with Humanoid")
    parser.add_argument("--save-video", type=str, help="Path to save video (e.g., 'video.mp4')", default=None)
    parser.add_argument("--duration", type=float, help="Duration of simulation in seconds", default=DEFAULT_DURATION)
    parser.add_argument("--make-plots", action="store_true", help="Enable real-time plotting", default=True)

    return parser.parse_args()


def load_humanoid_model() -> tuple[mujoco.MjModel, mujoco.MjData, int]:
    """Load the humanoid model and create data.

    Returns:
        Tuple containing the loaded model, data, and torso body ID

    Raises:
        FileNotFoundError: If the humanoid model could not be found
    """
    # Find the humanoid model in the test assets
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    xml_path = repo_root / "tests" / "assets" / "humanoid.xml"

    if not xml_path.exists():
        raise FileNotFoundError(f"Could not find humanoid model at {xml_path}")

    # Load the MuJoCo model and create data
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    # Get the body ID for the torso
    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")

    return model, data, torso_id


def setup_simulation_parameters(model: mujoco.MjModel, duration: float) -> tuple[np.ndarray, int, int]:
    """Setup simulation parameters.

    Args:
        model: The MuJoCo model
        duration: Duration of simulation in seconds

    Returns:
        Tuple containing control range, steps per control, and total steps
    """
    # Calculate steps based on physics and control timesteps
    physics_dt = model.opt.timestep  # Physics timestep from the model
    steps_per_control = int(CONTROL_DT / physics_dt)
    total_steps = int(duration / CONTROL_DT)

    # Define control limits based on the model
    ctrl_range = model.actuator_ctrlrange.copy()

    return ctrl_range, steps_per_control, total_steps


def setup_plotting_data(model: mujoco.MjModel) -> dict[int, str]:
    """Setup data for plotting.

    Args:
        model: The MuJoCo model

    Returns:
        Dictionary mapping actuator indices to names
    """
    # Get joint names for plotting
    actuator_names = []
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if name:
            actuator_names.append(name)
        else:
            actuator_names.append(f"actuator_{i}")

    # Create a mapping from actuator index to name for plots
    return {i: name for i, name in enumerate(actuator_names)}


def setup_viewer_plots(viewer: MujocoViewerHandler, actuator_mapping: dict[int, str]) -> None:
    """Setup plots in the viewer.

    Args:
        viewer: The MuJoCo viewer
        actuator_mapping: Mapping from actuator indices to names
    """
    # Add a plot group for actuator values
    viewer.add_plot_group(title="Actuator Commands", index_mapping=actuator_mapping, y_axis_min=-1.0, y_axis_max=1.0)

    # Add a plot group for joint positions (first 6 joints only for clarity)
    joint_pos_mapping = {i: f"Joint {i} Pos" for i in range(6)}
    viewer.add_plot_group(title="Joint Positions", index_mapping=joint_pos_mapping, y_axis_min=-2.0, y_axis_max=2.0)

    # Add a plot group for joint velocities
    joint_vel_mapping = {i: f"Joint {i} Vel" for i in range(6)}
    viewer.add_plot_group(title="Joint Velocities", index_mapping=joint_vel_mapping, y_axis_min=-5.0, y_axis_max=5.0)

    # Add custom plots for center of mass position and velocity
    viewer.add_plot("COM Height", y_label="Height (m)", y_axis_min=0.0, y_axis_max=2.0, group="COM")
    viewer.add_plot("COM Velocity", y_label="Velocity (m/s)", y_axis_min=-2.0, y_axis_max=2.0, group="COM")


def add_markers(
    viewer: MujocoViewerHandler,
    com_pos: np.ndarray,
    com_vel: np.ndarray,
    foot_contacts: tuple[bool, bool] = (False, False),
) -> None:
    """Add tracking markers to the scene.

    Args:
        viewer: The MuJoCo viewer
        com_pos: Center of mass position (x, y, z)
        com_vel: Center of mass velocity (x, y, z)
        foot_contacts: Tuple of (right_foot_contact, left_foot_contact) booleans
    """
    # 1. Add a stationary reference marker at the origin
    viewer.add_marker(
        name="origin",
        pos=np.array([0, 0, 0]),
        scale=np.array([0.1, 0.1, 0.1]),
        color=np.array([1, 1, 1, 0.5]),
        label="Origin",
    )

    # 2. A marker at the head
    viewer.add_marker(
        name="head_marker",
        scale=np.array([0.1, 0.1, 0.1]),
        color=np.array([0, 1, 0, 0.8]),
        label="Head",
        track_body_name="head",
    )

    # 3. A marker at the torso with velocity arrow
    torso_vel_magnitude = np.linalg.norm(com_vel[:2])
    if torso_vel_magnitude > 0.1:
        # Create velocity arrow when there's significant movement
        viewer.add_velocity_arrow(
            command_velocity=float(torso_vel_magnitude),
            base_pos=(0, 0, 1.7),
            scale=0.2,
            rgba=(1.0, 0.0, 0.0, 0.8),
            direction=[com_vel[0] / torso_vel_magnitude, com_vel[1] / torso_vel_magnitude, 0.0],
            label=f"Vel: {torso_vel_magnitude:.2f}",
        )

    # 4. Markers for the feet
    viewer.add_marker(
        name="right_foot",
        scale=np.array([0.05, 0.05, 0.05]),
        color=np.array([0, 0, 1, 0.8]),
        label="R Foot",
        track_body_name="foot_right",
    )

    viewer.add_marker(
        name="left_foot",
        scale=np.array([0.05, 0.05, 0.05]),
        color=np.array([1, 0, 0, 0.8]),
        label="L Foot",
        track_body_name="foot_left",
    )

    # 5. Add ground contact indicators for feet
    right_foot_contact, left_foot_contact = foot_contacts

    if right_foot_contact:
        viewer.add_marker(
            name="right_foot_contact",
            scale=np.array([0.1, 0.1, 0.01]),
            color=np.array([0, 1, 0, 0.8]),  # Green = contact
            track_body_name="foot_right",
            tracking_offset=np.array([0, 0, -0.05]),
            geom=mujoco.mjtGeom.mjGEOM_CYLINDER,
        )

    if left_foot_contact:
        viewer.add_marker(
            name="left_foot_contact",
            scale=np.array([0.1, 0.1, 0.01]),
            color=np.array([0, 1, 0, 0.8]),  # Green = contact
            track_body_name="foot_left",
            tracking_offset=np.array([0, 0, -0.05]),
            geom=mujoco.mjtGeom.mjGEOM_CYLINDER,
        )


def update_plots(viewer: MujocoViewerHandler, data: mujoco.MjData, com_pos: np.ndarray, com_vel: np.ndarray) -> None:
    """Update plots with current simulation data.

    Args:
        viewer: The MuJoCo viewer
        data: The MuJoCo data
        com_pos: Center of mass position (x, y, z)
        com_vel: Center of mass velocity (x, y, z)
    """
    # Update actuator command plots
    viewer.update_plot_group("Actuator Commands", data.ctrl.tolist())

    # Update joint position plots (first 6 joints)
    viewer.update_plot_group("Joint Positions", data.qpos[:6].tolist())

    # Update joint velocity plots
    viewer.update_plot_group("Joint Velocities", data.qvel[:6].tolist())

    # Update COM plots
    viewer.update_plot("COM Height", float(com_pos[2]))
    viewer.update_plot("COM Velocity", float(np.linalg.norm(com_vel)))


def run_simulation(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ctrl_range: np.ndarray,
    steps_per_control: int,
    total_steps: int,
    controller: SimulationController,
    viewer: MujocoViewerHandler,
    make_plots: bool,
) -> None:
    """Run the simulation loop.

    Args:
        model: The MuJoCo model
        data: The MuJoCo data
        ctrl_range: Control range for actuators
        steps_per_control: Number of physics steps per control step
        total_steps: Total number of control steps to run
        controller: The simulation controller for keyboard input
        viewer: The MuJoCo viewer
        make_plots: Whether to update plots
    """
    for step in range(total_steps):
        # If paused, only update the viewer and skip physics
        if controller.paused:
            viewer.update_and_sync()
            time.sleep(0.01)  # Small delay to prevent high CPU usage when paused
            continue

        # Generate random control signals
        random_action = np.random.uniform(low=ctrl_range[:, 0], high=ctrl_range[:, 1])

        # Apply the action
        data.ctrl[:] = random_action

        # Step the simulation forward one control step
        for _ in range(steps_per_control):
            mujoco.mj_step(model, data)

        # Get center of mass position and velocity
        com_pos = data.qpos[:3].copy()
        com_vel = data.qvel[:3].copy()

        # Add markers (visual elements for the viewer)
        add_markers(viewer, com_pos, com_vel)

        # Update plots with the current data if enabled
        if make_plots:
            update_plots(viewer, data, com_pos, com_vel)

        # Update visualization
        viewer.update_and_sync()

        # Add a small delay to make the visualization smoother
        time.sleep(max(0, CONTROL_DT - (model.opt.timestep * steps_per_control)))


def print_simulation_info(duration: float, save_video: str | None, make_plots: bool) -> None:
    """Print information about the simulation setup.

    Args:
        duration: Duration of simulation in seconds
        save_video: Path to save video or None
        make_plots: Whether plots are enabled
    """
    logger.info("Running simulation for %s seconds with control timestep %ss", duration, CONTROL_DT)
    logger.info("Video saving: %s", "Enabled" if save_video else "Disabled")
    logger.info("Plotting: %s", "Enabled" if make_plots else "Disabled")
    logger.info("Controls:")
    logger.info("  - Space: Pause/Resume simulation")
    logger.info("  - Esc: Exit simulation")


def main() -> None:
    """Main function to run the humanoid simulation example."""
    colorlogging.configure(level=logging.INFO)
    # Parse command line arguments
    args = parse_arguments()

    try:
        # Load model and get simulation parameters
        model, data, torso_id = load_humanoid_model()

        # Setup simulation parameters
        ctrl_range, steps_per_control, total_steps = setup_simulation_parameters(model, args.duration)

        # Setup plotting data
        actuator_mapping = setup_plotting_data(model)

        # Print simulation info
        print_simulation_info(args.duration, args.save_video, args.make_plots)
        logger.info("Torso body ID: %s", torso_id)

        # Create controller and start listener
        controller = SimulationController()
        # Start the listener but no need to keep a reference as it's stored in the controller
        controller.start_listener()

        # Run the passive viewer with our model
        with launch_passive(
            model,
            data,
            show_left_ui=True,
            show_right_ui=True,
            capture_pixels=args.save_video is not None,
            save_path=args.save_video,
            render_width=DEFAULT_RENDER_WIDTH,
            render_height=DEFAULT_RENDER_HEIGHT,
            ctrl_dt=CONTROL_DT,
            make_plots=args.make_plots,
        ) as viewer:
            # Setup the camera for a good view of the humanoid
            viewer.setup_camera(
                render_distance=4.0,
                render_azimuth=120.0,
                render_elevation=-20.0,
                render_lookat=[0.0, 0.0, 1.0],
                render_track_body_id=torso_id,  # Track the torso
            )

            # Setup plots if enabled
            if args.make_plots:
                setup_viewer_plots(viewer, actuator_mapping)

            # Run the simulation
            run_simulation(model, data, ctrl_range, steps_per_control, total_steps, controller, viewer, args.make_plots)

            # Cleanup the keyboard listener
            controller.stop_listener()

            logger.info("Simulation complete!")

    except FileNotFoundError as e:
        logger.error("Error: %s", e)
        sys.exit(1)
    except Exception as e:
        logger.error("Unexpected error: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
