# cell-flow-sims

Cell mechanics simulations with a fast periodic 2D testing path and the
existing spherical backend. The planar engine preserves the current model's
state-dependent radii, motility, adhesion, regularized repulsion, CIL
relaxation, and rotational diffusion while removing spherical geometry from
the experiment loop.

## Installation

For the numerical backend and tests only:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"
```

Add the lightweight 2D sandbox without installing Qt or Napari:

```bash
pip install -e ".[test,sandbox]"
```

The existing 3D viewer remains available as an optional extra:

```bash
pip install -e ".[napari]"
napari
```

Alternatively, `environment.yml` creates the complete development environment
with the backend, Matplotlib, Napari, Jupyter, and tests:

```bash
conda env create -f environment.yml
conda activate cell-flow-sims
```

## Periodic 2D sandbox

Start the interactive Matplotlib sandbox:

```bash
python examples/run_2d_sandbox.py
```

It provides run/pause, single-step, deterministic reset, live collective
metrics, polarity arrows, and readable controls for Cell Count, Random Seed,
Motility Force (`Fm`), Rotational Diffusion (`Dr`), CIL Rate (`fcil`), Adhesion
Strength (`w`), Repulsion Strength (`k_rep`), Hard-Core Ratio (`alpha_dmin`),
Time Step, Initial Clearance, Steps Per Frame, and each state's Division Rate
(`lambda_div`) and Division Pause (`tau_div`). Division controls update live;
cell count, seed, and clearance apply on reset. Load a different YAML file with
`--config path/to/config.yaml`.

After daughter insertion, the overlap-connected component reached from the
daughters is projected instantaneously to non-overlapping disk positions. The
GUI reports the number of shoved cells and maximum event displacement. This
correction is not included in cell velocity; see `docs/PLANAR_2D.md` for its
algorithm and interpretation.

Run the same backend without opening a window or importing Matplotlib:

```bash
python examples/run_2d_sandbox.py --headless --steps 100
```

For a reproducible data-loading workflow with two cell states, open
`notebooks/two_state_data_simulation.ipynb`. It loads the bundled synthetic CSV,
provides one editable parameter cell (including the two state-specific
proliferation rates), runs the planar engine, and plots the initial/final
populations and collective diagnostics.

The command prints final diagnostics as JSON and exits. The documented default
configuration is in `configs/sim_2d.yaml`.

The engine is also usable directly:

```python
import numpy as np

from cell_sphere_sim.planar import (
    PlanarParams,
    PlanarSimulationEngine,
    init_random_periodic,
)
from cell_sphere_sim.state import StateTable

rng = np.random.default_rng(123)
states = StateTable(
    R=np.array([0.35]),
    Fm=np.array([1.0]),
    Dr=np.array([0.05]),
    fcil=np.array([2.0]),
    w=np.array([0.2]),
    lambda_div=np.array([0.0]),
    tau_div=np.array([1.0]),
)
state_id = np.zeros(500, dtype=np.int32)
x, p = init_random_periodic(
    500,
    (32.0, 32.0),
    state_id,
    states,
    rng,
    initial_min_separation_factor=0.9,
)
engine = PlanarSimulationEngine(
    x,
    p,
    state_id,
    np.zeros((500, 0)),
    states,
    PlanarParams(
        box_size=(32.0, 32.0),
        gamma_s=1.0,
        k_rep=2.0,
        alpha_dmin=0.2,
        eps=1e-8,
        dt=0.01,
    ),
    rng=rng,
)
diagnostics = engine.run(100)
print(diagnostics[-1])
```

Positions are always wrapped into `[0, Lx) x [0, Ly)`. Neighbor detection,
contact distances, directions, and forces all use the rectangular minimum-image
convention. An unwrapped trajectory is retained for mean-squared displacement.

Two similarly named parameters have intentionally different roles:

- `alpha_dmin` regularizes the existing force law near its hard core.
- `initial_min_separation_factor` controls initial packing clearance only. Its
  conservative default of `0.9` avoids initializing near the force singularity.

See `docs/PLANAR_2D.md` for equations, diagnostic definitions, and conventions,
and `docs/PARAMETER_SCALING.md` for dimensions, nondimensional groups, and
calibration caveats.

## Continuum workbench

The continuum route is a separate periodic-field workbench beside the discrete
cell sandbox. It includes Passive Model B, Active Model B, a mechanistic
density–polarization MIPS closure, and Keller–Segel autochemotaxis:

```bash
conda activate cell-flow-sims
python examples/run_continuum_workbench.py
```

The interactive view provides model/preset selection, non-overlapping short
parameter labels, run/pause/step/reset controls, live primary and secondary
fields, vector overlays where applicable, diagnostics, and reproducible export.
Numerical parameter changes affect the next integration step. Signal mode and
other controls that change field semantics reset the run. The Setup dialog
contains reset-required domain, seed, initial-noise, and droplet controls.

Run without Matplotlib or a display:

```bash
python examples/run_continuum_workbench.py \
  --headless --model model_b --preset spinodal --steps 1000
```

Two-parameter/seed sweeps are available through
`examples/run_continuum_sweep.py`; its no-argument defaults form a small
mean-density/free-speed MIPS operational map. The default workbench and sweep
configurations are `configs/continuum.yaml` and
`configs/continuum_mips_sweep.yaml`.

The complete downloadable model audit—including equations, assumptions,
parameter scaling relations, initialization from measured fields, numerics,
limitations, validation, and primary references—is in
[`docs/CONTINUUM_MODELS.md`](docs/CONTINUUM_MODELS.md).

## Spherical backend and Napari UI

The original spherical example remains available:

```bash
python examples/run_minimal.py
```

With the `napari` extra installed, launch Napari and select
Plugins -> Cell Sphere Sim. The viewer renders spherical cells, the context
surface, tracks in `(track_id, t, z, y, x)` format, and live diagnostics.

## Tests

Some environments expose incompatible third-party Napari or Numba pytest
plugins. Backend validation can disable unrelated plugin auto-loading:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
```

## Current limitations

Planar division is an unrestricted state-specific Poisson birth process: it has
no cell-cycle stages, growth, death, crowding suppression, or population cap.
The planar engine has no PDE field coupling and does not add alignment or
change any biological interaction law. Its force regularization is not yet a
geometric non-crossing constraint during ordinary mechanics steps; only
division insertion has event-local projection. The vector-based CIL relaxation
retains the spherical implementation's narrow exact-antipode degeneracy. The
existing Pandas track store is 3D-specific; the dimension-agnostic
`TrajectoryStore` works with variable-length planar arrays and lineage IDs.
The continuum models are coarse-grained periodic PDEs and are not coupled to the
discrete planar or spherical cell engines.
