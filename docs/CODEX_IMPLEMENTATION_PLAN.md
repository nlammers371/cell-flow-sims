# CODEX implementation plan (Phase 1-focused)

> Goal: provide a deterministic, tested simulation engine for cell dynamics on a sphere (mechanics + 3D polarity + CIL + optional division), with clean interfaces for future PDE fields, HT sweeps, and a napari GUI.

This doc is the contract for Codex: module layout, APIs, shapes, and tests.

---

## 1) Repo layout (proposed)

```
cell_sphere_sim/
  __init__.py
  engine.py
  config.py
  neighbors.py
  forces.py
  polarity.py
  division.py
  state.py
  fields/
    __init__.py
    base.py
  io/
    __init__.py
    tracks_init.py
    stores.py
  metrics.py
tests/
  test_constraints.py
  test_transport.py
  test_cil.py
  test_division.py
  test_neighbors.py
examples/
  run_minimal.py
  init_from_tracks.py
configs/
  sim_minimal.yaml
  states_minimal.yaml
pyproject.toml  (or setup.cfg)
README.md
```

Phase 1 does **not** need napari plugin code yet. It must provide data structures that a napari layer can consume.

---

## 2) Data shapes and types (strict)

- N cells
- positions: `x: (N,3) float32/float64`
- polarity: `p: (N,3) float32/float64` (unit, tangent)
- normal: `n = x / R_E` (computed on the fly)
- discrete state ids: `state_id: (N,) int32`
- continuous state vars: `state_vars: (N,M) float32/float64` (M can be 0)
- per-state tables: arrays of shape `(K,)` float
- RNG: numpy Generator passed explicitly; no global RNG

All functions should accept and return numpy arrays; avoid pandas in core engine.

---

## 3) Core classes and signatures

### 3.1 State parameter table

`cell_sphere_sim/state.py`

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class StateTable:
    # required per-state params
    R: np.ndarray          # (K,)
    Fm: np.ndarray         # (K,)
    Dr: np.ndarray         # (K,)
    fcil: np.ndarray       # (K,)
    w: np.ndarray          # (K,)
    lambda_div: np.ndarray # (K,)  (can be zeros if disabled)
    tau_div: np.ndarray    # (K,)  (seconds or time units)
```

### 3.2 Field interface + NullField stub

`cell_sphere_sim/fields/base.py`

```python
from typing import Protocol
import numpy as np

class FieldModel(Protocol):
    C: int
    def sample(self, x: np.ndarray, t: float) -> np.ndarray: ...
    def reset_sources(self) -> None: ...
    def accumulate_sources(self, x: np.ndarray, sources: np.ndarray) -> None: ...
    def step(self, dt: float) -> None: ...

class NullField:
    def __init__(self, C: int = 2): ...
    def sample(self, x, t): return np.zeros((len(x), self.C), dtype=x.dtype)
    def reset_sources(self): ...
    def accumulate_sources(self, x, sources): ...
    def step(self, dt): ...
```

### 3.3 Cell update hooks (field/state coupling)

`cell_sphere_sim/engine.py` expects callables:

```python
from typing import Callable, NamedTuple

class BehaviorParams(NamedTuple):
    # per-cell effective parameters
    R: np.ndarray
    Fm: np.ndarray
    Dr: np.ndarray
    fcil: np.ndarray
    w: np.ndarray
    lambda_div: np.ndarray
    tau_div: np.ndarray

CellUpdateFn = Callable[
  [np.ndarray, np.ndarray, np.ndarray, dict, float, np.random.Generator, StateTable],
  tuple[np.ndarray, np.ndarray, BehaviorParams]
]
# args: state_id, state_vars, fields, contact_metrics, dt, rng, state_table

CellSourcesFn = Callable[
  [np.ndarray, np.ndarray, np.ndarray, float, StateTable],
  np.ndarray
]
# returns sources (N,C)
```

Provide defaults:
- `default_cell_update`: identity on states; uses pure table lookup for behavior params
- `default_cell_sources`: zeros

### 3.4 Neighbor list

`cell_sphere_sim/neighbors.py`

```python
def build_spatial_hash(x: np.ndarray, cell_size: float) -> dict[tuple[int,int,int], np.ndarray]: ...

def candidate_pairs_from_hash(x: np.ndarray, hash_map: dict, cell_size: float) -> list[tuple[int,int]]: ...
```

Phase 1: correctness over micro-optimizations. Must not miss contacts.

### 3.5 Forces + contact metrics

`cell_sphere_sim/forces.py`

```python
@dataclass
class ContactMetrics:
    contact_count: np.ndarray      # (N,) int
    contact_dir_sum: np.ndarray    # (N,3) float (sum of tangent-to-neighbor unit vectors)

def compute_contact_forces_and_metrics(
    x: np.ndarray,
    state_id: np.ndarray,
    behavior: BehaviorParams,
    k_rep: float,
    alpha_dmin: float,
    eps: float,
    candidate_pairs: list[tuple[int,int]],
    R_E: float
) -> tuple[np.ndarray, ContactMetrics]:
    # returns F_contact (N,3)
    ...
```

`contact_dir_sum[i]` must be computed using tangent projection at `n_i`.

### 3.6 Polarity utilities

`cell_sphere_sim/polarity.py`

```python
def parallel_transport(p: np.ndarray, n: np.ndarray, n_new: np.ndarray, eps: float) -> np.ndarray: ...

def random_tangent_rotation(p: np.ndarray, n: np.ndarray, delta: float) -> np.ndarray: ...

def cil_target_flee(contact_dir_sum: np.ndarray, n_new: np.ndarray, eps: float) -> tuple[np.ndarray, np.ndarray]:
    # returns (has_target (N,) bool, p_flee (N,3))
    ...
```

### 3.7 Division

`cell_sphere_sim/division.py`

```python
def sample_divisions(lambda_div: np.ndarray, dt: float, rng: np.random.Generator) -> np.ndarray:
    # returns boolean mask (N,)
    ...

def apply_divisions(
    x: np.ndarray,
    p: np.ndarray,
    state_id: np.ndarray,
    state_vars: np.ndarray,
    paused_until: np.ndarray,
    t: float,
    behavior: BehaviorParams,
    R_E: float,
    split_scale: float,
    rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # inserts new cells, returns updated arrays
    # daughters inherit state_id and state_vars
    # division axis is random tangent direction
    ...
```

`paused_until` is shape `(N,)` float.

### 3.8 Simulation engine

`cell_sphere_sim/engine.py`

```python
@dataclass
class SimParams:
    R_E: float
    gamma_s: float
    k_rep: float
    alpha_dmin: float
    eps: float
    dt: float
    neighbor_cell_size: float
    record_interval: int
    division_enabled: bool = True
    split_scale: float = 0.5   # scale factor for daughter separation vs sigma (tunable)
    relax_substeps: int = 5
    relax_dt_scale: float = 0.2

class SimulationEngine:
    def __init__(
        self,
        x: np.ndarray,
        p: np.ndarray,
        state_id: np.ndarray,
        state_vars: np.ndarray,
        state_table: StateTable,
        params: SimParams,
        field: FieldModel | None = None,
        cell_update: CellUpdateFn | None = None,
        cell_sources: CellSourcesFn | None = None,
        rng: np.random.Generator | None = None
    ): ...

    def step(self, t: float) -> dict:
        # returns diagnostics dict for logging
        ...

    def run(self, n_steps: int, t0: float = 0.0, store=None, callback=None) -> None: ...
```

Mechanics per step:

1) fields = field.sample(x,t) (or zeros)
2) (state_id, state_vars, behavior) = cell_update(...)
3) candidate pairs via neighbors
4) (F_contact, metrics) = forces(...)
5) motility gate: gate = (t >= paused_until)
6) explicit x update + projection
7) transport p
8) compute CIL targets; do relax+diffusion update on p
9) sample/apply divisions (if enabled), including motility pause
10) compute sources and advance field (optional, NullField OK)

---

## 4) Initialization utilities (Phase 1)

`cell_sphere_sim/io/tracks_init.py`

- `init_from_napari_tracks(df: pd.DataFrame, t0: int|float, R_E: float, ...) -> x,p,state_id,state_vars`

Assume napari tracks DataFrame columns:
- required: `track_id`, `t`, `z`, `y`, `x` (3D) or `track_id`, `t`, `y`, `x` (2D)
- optional: `state_id` or categorical label; optional state_vars columns

Initialize polarity:
- if next frame exists per track: p from finite difference velocity projected to tangent and normalized
- else random tangent direction

Random init can be in `examples/` in Phase 1.

---

## 5) Tests (must pass)

### 5.1 Constraints
`test_constraints.py`
- after many steps: max | ||x||-R_E | < 1e-5
- max |p·n| < 1e-6
- max | ||p||-1 | < 1e-6

### 5.2 Transport sanity
`test_transport.py`
- small move: transported p remains close to original (no NaNs, no flips)

### 5.3 CIL behavior
`test_cil.py`
- 2 cells in contact: computed p_flee points away from neighbor (dot(p_flee, tangent_to_neighbor) < 0)
- with fcil>0 and Dr=0: p moves toward p_flee over steps

### 5.4 Division
`test_division.py`
- division inserts exactly one new cell
- daughters inherit state_id and state_vars
- paused_until set for both daughters >= t
- during pause: motility term is zero (velocity arises only from contact forces)

### 5.5 Neighbor correctness
`test_neighbors.py`
- brute force contacts vs neighbor candidates: all true contacts appear in candidates for random configurations at N~200

---

## 6) Phase 1 deliverables

- Engine runs a minimal example with N~1000 on sphere
- Tests pass via `pytest`
- `examples/run_minimal.py` produces a short run and prints diagnostics
- `examples/init_from_tracks.py` shows tracks init path (can use a synthetic df)

---

## 7) Phase 2+ notes (not for Phase 1 implementation)

- napari plugin (Qt dock widget) consumes the engine via a ring buffer store
- sphere mesh + density shading can be done by mapping scalar values to mesh vertex colors
- HT sweeps runner (multiprocessing) calls the engine and writes metrics/trajectories

