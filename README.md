# kscale-mujoco-viewer

Mujoco viewer maintained by K-Scale Labs.

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
import mujoco, time
from kmv.app.viewer import QtViewer

model  = mujoco.MjModel.from_xml_path("path/to/model.xml")
data   = mujoco.MjData(model)
viewer = QtViewer(model)

while viewer.is_open:
    mujoco.mj_step(model, data)

    # send state
    viewer.push_state(data.qpos, data.qvel, sim_time=data.time)

    # send a few scalars to the “Debug” plot group
    viewer.push_plot_metrics(
        scalars={
            "qpos0": float(data.qpos[0]),
            "qvel0": float(data.qvel[0]),
        },
        group="Debug",
    )

    time.sleep(model.opt.timestep)

viewer.close()
```

# Acknowledgements

Originally referenced from [mujoco-python-viewer](https://github.com/gaolongsen/mujoco-python-viewer).