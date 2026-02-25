# cell-flow-sims

Phase 1 implementation of a cell dynamics simulator on a sphere.

## Quick start (conda, recommended)

All dependencies (backend + tests + GUI) are captured in a single conda file.

Note: napari is not yet compatible with Python 3.14. The conda file pins a
working Python version for you.

```bash
conda env create -f environment.yml
conda activate cell-flow-sims
pytest
python examples/run_minimal.py
```

## Quick start (venv, backend only)

Use this if you only need the backend simulation and tests. The virtual
environment keeps dependencies isolated from your system Python and does
not install the napari GUI stack.

```bash
python -m venv .venv
source .venv/bin/activate
# Install the package in editable mode so local changes take effect immediately.
pip install -e .
# Install the test runner.
pip install pytest
pytest
python examples/run_minimal.py
```

## Example: simple simulation from the command line

```bash
python examples/run_minimal.py
```

This prints per-step diagnostics to stdout. It does not write files by default.

## Example: save results and load in Jupyter

The minimal example prints diagnostics only. If you want to save output, add a
small writer to your script and then load it in a notebook. For example:

```python
import numpy as np

# ... after running your sim loop
np.savez(
	"outputs/minimal_run.npz",
	x=engine.x,
	p=engine.p,
	state_id=engine.state_id,
	state_vars=engine.state_vars,
)
```

Then in a notebook:

```python
import numpy as np

data = np.load("outputs/minimal_run.npz")
x = data["x"]
p = data["p"]
state_id = data["state_id"]
state_vars = data["state_vars"]
```

## Notebook workflow (params -> run -> save -> analyze)

See [notebooks/workflow_demo.ipynb](notebooks/workflow_demo.ipynb) for a
step-by-step workflow that:

- sets model parameters
- initializes and runs a simulation
- saves results to disk
- performs a few simple analyses

## Napari UI

```bash
napari
```

In napari: Plugins -> Cell Sphere Sim.
