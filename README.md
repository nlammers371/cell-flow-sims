# cell-flow-sims

Phase 1 implementation of a cell dynamics simulator on a sphere.

## Quick start (conda, recommended)

All dependencies (backend + tests + GUI) are captured in a single conda file.

```bash
conda env create -f environment.yml
conda activate cell-flow-sims
pytest
python examples/run_minimal.py
```

## Quick start (venv, backend only)

If you only need the backend sims and tests, a plain venv is enough.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
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

## Napari UI

```bash
napari
```

In napari: Plugins -> Cell Sphere Sim.
