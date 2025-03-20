# kscale-mujoco-viewer

Mujoco viewer maintained by K-Scale Labs.

Originally referenced from [mujoco-python-viewer](https://github.com/gaolongsen/mujoco-python-viewer).

## Installation

```bash
pip install kmv
```

## Usage

```python
from kmv.viewer import launch_passive

with launch_passive(viewer_model, viewer_data) as viewer:
    while doing_stuff:
        # do stuff
        viewer.update_and_sync()
```

