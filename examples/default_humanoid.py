#!/usr/bin/env python3
"""Minimal example demonstrating the K-Scale MuJoCo Viewer with the default humanoid model.

Loads the humanoid and lets it fall down under gravity - no control, just raw performance testing.
"""

import argparse
import time
import mujoco
from pathlib import Path

from kmv.app.viewer import QtViewer


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="K-Scale MuJoCo Viewer example with humanoid model")
    parser.add_argument(
        "--disable-plots", 
        action="store_true",
        help="Disable real-time scalar plots in the viewer"
    )
    args = parser.parse_args()
    
    # Invert the disable flag to get enable flag
    enable_plots = not args.disable_plots
    
    # Load the humanoid model
    xml_path = Path(__file__).parent.parent / "tests" / "assets" / "humanoid.xml"
    
    print(f"Loading model from: {xml_path}")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    
    # Create the viewer with plots enabled based on command line argument
    viewer = QtViewer(model, data, width=1200, height=800, enable_plots=enable_plots)
    
    # Set camera view
    viewer.cam.distance = 3.5
    viewer.cam.azimuth = 90.0
    viewer.cam.elevation = -10.0
    viewer.cam.lookat[:] = [0.0, 0.0, 1.0]
    
    print("Starting simulation... (Press Q or Esc to quit)")
    if enable_plots:
        print("Real-time plots enabled")
    else:
        print("Real-time plots disabled")
    
    # Simple timing for FPS
    step_count = 0
    last_time = time.time()
    
    try:
        while True:
            # Step physics
            mujoco.mj_step(model, data)
            
            # Update viewer
            viewer.push_mujoco_frame(
                qpos=data.qpos.copy(),
                qvel=data.qvel.copy(),
                sim_time=data.time
            )
            
            # Push scalar data for plotting only if plots are enabled
            if enable_plots:
                scalars = {
                    "qpos_0": float(data.qpos[0]),
                    "qpos_1": float(data.qpos[1]),
                    "qpos_2": float(data.qpos[2]),
                    "qvel_0": float(data.qvel[0]),
                    "qvel_1": float(data.qvel[1]),
                    "qvel_2": float(data.qvel[2]),
                }
                viewer.push_scalar(data.time, scalars)
            
            viewer.update()
            
            step_count += 1
            
            # Print FPS every 500 steps
            if step_count % 500 == 0:
                current_time = time.time()
                elapsed = current_time - last_time
                fps = 500.0 / elapsed
                print(f"Step {step_count}: {fps:.1f} FPS")
                last_time = current_time
                
    except KeyboardInterrupt:
        print("\nDone!")


if __name__ == "__main__":
    main()
