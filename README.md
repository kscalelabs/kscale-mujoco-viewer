# kscale-mujoco-viewer

Mujoco viewer maintained by K-Scale Labs.

Originally referenced from [mujoco-python-viewer](https://github.com/gaolongsen/mujoco-python-viewer).

## Installation

```bash
pip install kmv
```

## Examples

Run the humanoid example script to see the viewer in action:

```bash
# cd to the root of the repo
python examples/default_humanoid.py
```


## Usage

```python
import mujoco
from kmv.viewer import launch_passive

# Load model and create data
model = mujoco.MjModel.from_xml_path("path/to/model.xml")
data = mujoco.MjData(model)

# Run viewer
with launch_passive(model, data, make_plots=True) as viewer:
    # Setup camera and other options if needed
    viewer.setup_camera(render_distance=4.0, render_lookat=[0.0, 0.0, 1.0])
    
    while running_simulation:
        # Update physics
        mujoco.mj_step(model, data)
        
        # Update visualization
        viewer.update_and_sync()
```

