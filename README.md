# kscale-mujoco-viewer

Mujoco viewer maintained by K-Scale Labs.

https://github.com/user-attachments/assets/5312fa03-7215-4c1b-9d66-acaa18b96a3c

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

# Troubleshooting

If you have issues with linux machines that have both integrated and nvidia GPUs, you may need to set these environment variables to force use of the nvidia GPU.

```bash
__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia python examples/default_humanoid.py
```


# Acknowledgements

Originally referenced from [mujoco-python-viewer](https://github.com/gaolongsen/mujoco-python-viewer).
