# IMPLEMENTATION_PLAN_v2.md — Backend Revision 2 workplan (Codex contract)

This is the actionable plan for Revision 2. It assumes an existing repo matching Phase 1 layout (engine/forces/neighbors/polarity/division/state).

---

## 1) High-level tasks

T1. Replace neighbor candidate generation with SciPy cKDTree (return index arrays).  
T2. Vectorize contact forces + CIL metrics (no Python loops over pairs).  
T3. Add `dt` default + guardrails/warnings.  
T4. Add robust initialization machinery:
- no overlaps within `d_min`
- anisotropy controls
- state composition + mixing controls  
T5. Add progress bar in `run`.  
T6. Add/Update diagnostics notebook (1–3 runs, plots, constraint checks).  
T7. Add code overview doc with class attribute meanings and module responsibilities.  
T8. Add pandas tracks output store (primary output format).

---

## 2) Required new/updated modules

### 2.1 neighbors.py (replace)

Create:

```python
def candidate_pairs_ckdtree(x: np.ndarray, r: float) -> tuple[np.ndarray, np.ndarray]:
    # returns i_idx, j_idx with i_idx<j_idx, dtype int32
```

Add a helper to compute `r`:

```python
def interaction_radius(behavior_R: np.ndarray, buffer: float) -> float:
    # sigma_max = 2*max(R)
    # return sigma_max*(1+buffer)
```

Engine will call cKDTree every step (N<=5000, fine). Optional future optimization: reuse tree if needed.

Dependency: add `scipy` to requirements.

---

### 2.2 forces.py (vectorize)

Replace `compute_contact_forces_and_metrics(..., candidate_pairs: list[tuple])` with:

```python
def compute_contact_forces_and_metrics(
    x: np.ndarray,
    behavior: BehaviorParams,
    k_rep: float,
    alpha_dmin: float,
    eps: float,
    i_idx: np.ndarray,
    j_idx: np.ndarray,
    R_E: float,
) -> tuple[np.ndarray, ContactMetrics]:
```

Vectorized steps:

1) Gather pairwise vectors:

- `xi = x[i_idx]`, `xj = x[j_idx]`
- `dvec = xi - xj`
- `d = ||dvec||` (shape P)
- `sigma = R[i_idx] + R[j_idx]`
- contact mask `m = d < sigma`; filter all arrays to contacts only

2) Compute forces in bulk for contacts:

- `d_min = alpha_dmin*sigma`
- `d_eff = max(d, d_min+eps)` elementwise
- `n_hat = dvec / d_eff[:,None]`
- `rep`, `adh`, `f_mag`, `f_vec`

3) Scatter-add to forces:

- `np.add.at(F_contact, i, f_vec)`
- `np.add.at(F_contact, j, -f_vec)`

4) Contact counts:

- `np.add.at(contact_count, i, 1)`
- `np.add.at(contact_count, j, 1)`

5) CIL contact_dir_sum:
- compute normals `n = x/R_E` (N,3)
- for i-side:
  - `d_ij = xj - xi`
  - `n_i = n[i]`
  - `d_t = d_ij - (d_ij·n_i) n_i` (vectorized)
  - normalize and add with `np.add.at(contact_dir_sum, i, d_t_hat)`
- similarly j-side

Return `ContactMetrics(contact_count, contact_dir_sum)`.

No Python loops over pairs.

---

### 2.3 engine.py updates

#### dt defaults + warnings
Change `SimParams.dt: float` to `dt: float | None`.

Add:

- `compute_default_dt(state_table, gamma_s, eps, eta=0.02)`
- `_warn_dt(dt, state_table, gamma_s, eps)` using dx thresholds

In `__init__`, if dt None, compute default and store into `self.params.dt` (or build an internal dt).

#### neighbor radius parameter
Replace `neighbor_cell_size` with:

- `neighbor_radius_buffer: float` (e.g. 0.1) OR explicit `neighbor_radius: float | None`

Engine computes `R_query` from `behavior.R` each step (or from max state radius if behavior.R constant), then calls `candidate_pairs_ckdtree`.

#### progress bar
In `run(..., show_progress: bool = False)`:
- if show_progress: wrap `range(n_steps)` with `tqdm`
- avoid importing tqdm unless needed (optional dependency).

#### diagnostics keys
Return additional diagnostics:
- `n_candidates` (candidate pairs count)
- `n_contacts_pairs` (after filtering d<sigma)
- `min_d_contact` (min d among contacts, if any)
- constraint drift (optional)

---

### 2.4 init.py (new)

Create `cell_sphere_sim/init.py` with:

```python
def init_random_on_sphere(
    N: int,
    R_E: float,
    state_ids: np.ndarray,
    state_table: StateTable,
    alpha_dmin: float,
    eps: float,
    rng: np.random.Generator,
    pos_mode: str = "uniform",
    pos_strength: float = 0.0,
    mixing: float = 1.0,
    heading_mode: str = "isotropic",
    heading_strength: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    # returns x (N,3), p (N,3)
```

Requirements:
- uses rejection sampling with a KD-tree to ensure `d_ij >= d_min_ij` at init.
- supports simple anisotropy modes (as in MODEL_SPEC_v2).
- supports mixing control by controlling spatial assignment of state groups.

Also create:

```python
def sample_state_ids(N, composition, rng) -> np.ndarray:
    # composition can be dict{state_id: fraction} or list of counts
```

---

### 2.5 io/stores.py (new store for pandas tracks output)

Implement:

```python
class PandasTracksStore:
    def __init__(self): ...
    def append(t, x, v, state_id, track_id, extra=None): ...
    def to_dataframe() -> pandas.DataFrame: ...
```

Data format:
- columns: track_id, t, z,y,x, vz,vy,vx, state_id (+ extras)

Track identity:
- engine maintains `track_id: (N,) int64` as persistent identity
- on division, assign a new track_id to the new daughter; optionally record parent_id

Velocity:
- choose and document whether v is pre-update or post-update.
- recommended: compute v_post from `(x_new - x_old)/dt` after projection.

---

## 3) Division performance fix (recommended)

Replace current division implementation that repeatedly appends arrays and vstack/concatenate in a loop.

Plan:
- determine indices of dividing cells: `div_idx`
- allocate output arrays of size `N + len(div_idx)`
- fill:
  - mothers updated in-place in first N slots
  - daughters written into appended slots
- update track_id for daughters, paused_until for both

---

## 4) Diagnostics notebook deliverable

Add `notebooks/backend_diagnostics.ipynb` (or keep in /examples) that:
- runs sparse, 2-cell, and dense scenarios
- plots snapshots + time series diagnostics
- highlights candidate-pair counts and constraint drift

Notebook must import the installed package or local repo reliably.

---

## 5) Overview doc deliverable

Add `docs/backend_overview.md`:
- module map and responsibilities
- SimulationEngine attributes and meanings
- step order
- key parameters and interpretation
- output formats, especially pandas tracks
- common failure modes and tuning tips (dt, neighbor radius buffer)

---

## 6) Acceptance criteria / tests (extend existing)

Existing tests must pass. Add/extend:

- neighbor: cKDTree pairs radius correctness (no missed contacts for r >= sigma_max*(1+buf))
- forces: vectorized force equals previous loop version on a small random case (optional)
- init: ensure all pairwise d >= d_min (check with KD-tree query_pairs)
- tracks store: dataframe has required columns and correct row count

Also add a quick performance smoke test (not strict timing) to ensure N=3000 runs 50 steps without pathological candidate explosion.

