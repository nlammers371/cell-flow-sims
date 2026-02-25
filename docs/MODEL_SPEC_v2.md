# MODEL_SPEC_v2.md — Cell dynamics on a sphere (Revision 2)

This document supersedes `KF_cell_model_specs_UPDATED.md` for Revision 2. It preserves the core model but updates:

- Neighbor finding (**SciPy cKDTree**)
- Force + CIL metric computation (**vectorized**)
- `dt` guardrails + default `dt` heuristic
- Initialization constraints (no overlaps below `d_min`) + anisotropy/mixing controls
- Progress reporting
- Diagnostics notebook expectations
- Output data formats (**primary output supports pandas "tracks" format**)

Scope note: This spec focuses on the **backend** engine and initialization/diagnostics utilities. The napari UI remains a thin wrapper that consumes engine state.

---

## 0) Core state and geometry

- Sphere radius: `R_E`
- Position: `x_i ∈ R^3`, constrained to `||x_i|| = R_E`
- Normal: `n_i = x_i / R_E`
- Polarity: `p_i ∈ R^3`, constrained to:
  - `p_i · n_i = 0`
  - `||p_i|| = 1`
- Hybrid intrinsic state:
  - `state_id_i ∈ {0..K-1}` (K typically 1–3)
  - `state_vars_i ∈ R^M` (M≥0; M may be 0 in v1/v2)

Per-cell **behavior parameters** are produced each step by a user hook `cell_update(...)` returning a `BehaviorParams` bundle. Intended default is “Option A”: baseline by discrete state plus analytic modulation by fields/state_vars, but **the engine does not assume a multiplicative form**.

---

## 1) Contact geometry

- Chord distance: `d_ij = ||x_i - x_j||`
- Radii: `R_i = behavior.R[i]`
- Contact range: `sigma_ij = R_i + R_j`
- In contact if `d_ij < sigma_ij`

Hard-core:
- `d_min_ij = alpha_dmin * sigma_ij`
- In force eval: `d_eff = max(d_ij, d_min_ij + eps)`

---

## 2) Contact forces + CIL contact metrics

### 2.1 Forces (unchanged formulas)

For `d_ij < sigma_ij`:

- Repulsion:
  - `F_rep = k_rep * ((sigma - d_eff) / (d_eff - d_min))^(3/2)`
- Adhesion:
  - overlap `δ = sigma - d_eff`
  - `r_bar = 0.5 * (R_i + R_j)`
  - `F_adh = (w_i*w_j / r_bar) * δ`
- Net:
  - `F_mag = F_rep - F_adh`
  - `n_hat = (x_i - x_j)/d_eff`
  - `F_ij = F_mag * n_hat`
  - add `+F_ij` to i and `-F_ij` to j

### 2.2 CIL contact direction metric

For each contact pair (i,j), compute tangent unit direction **toward** neighbor:

- `d_ij = x_j - x_i`
- tangent at i: `d_t_i = d_ij - (d_ij·n_i)n_i`
- if `||d_t_i||>eps`: add `d_t_i/||d_t_i||` to `contact_dir_sum[i]`
- similarly for j

Also count contacts per cell: `contact_count[i]`.

### 2.3 Implementation constraint (Revision 2 requirement)

**Force and metric computation MUST be vectorized** over candidate pairs.
- Candidate pairs are provided as two integer arrays `i_idx`, `j_idx` (shape `(P,)`).
- Avoid Python loops over pairs.
- Accumulate with `np.add.at` (or equivalent scatter-add).

---

## 3) Neighbor candidates (Revision 2: SciPy cKDTree)

Replace spatial hash + Python set with SciPy `cKDTree`.

- Build `tree = scipy.spatial.cKDTree(x)`
- Get candidate pairs:
  - `pairs = tree.query_pairs(r=R_query, output_type="ndarray")` → `(P,2)` int array
  - set `i_idx = pairs[:,0]`, `j_idx = pairs[:,1]`

### 3.1 Query radius

Let:

- `R_max = max(behavior.R)`
- `sigma_max = 2 * R_max`
- choose buffer `buf ∈ [0.05, 0.15]`

Then:

- `R_query = sigma_max * (1 + buf)`

This must be large enough not to miss any true contacts.

---

## 4) Motility and overdamped position update (explicit Euler)

Per-cell motility magnitude: `Fm_i = behavior.Fm[i]` (may be gated off during division pause).

Velocity:

- `gate_i = 1 if t >= paused_until[i] else 0`
- `v_i = ( gate_i * Fm_i * p_i + F_contact_i ) / gamma_s`

Explicit Euler:

1. `x_tmp = x + dt * v`
2. project to sphere:
   - `x_new = R_E * x_tmp / ||x_tmp||`
3. `n_new = x_new / R_E`

---

## 5) Polarity update (3D tangent) — transport + CIL + diffusion

Unchanged from prior spec; key invariants must be enforced each step.

- Parallel transport via minimal rotation mapping `n -> n_new`
- CIL flee target from `contact_dir_sum`
- Exponential relaxation toward `p_flee` for contacting cells
- Rotational diffusion by random rotation about axis `n_new` with `δ ~ Normal(0, 2 Dr dt)`

---

## 6) Proliferation (optional, event-driven)

- Division probability per dt:
  - `P(divide) = 1 - exp(-lambda_div_i * dt)`
- On division:
  - daughters inherit `state_id` and `state_vars` exactly
  - random division axis uniformly sampled in tangent plane
  - place daughters by symmetric split, reproject
  - **motility pause**: gate motility off for duration `tau_div[state]`
  - optional short relaxation burst to resolve overlaps

Implementation constraint (Revision 2 recommendation):
- avoid repeated `vstack/concatenate` inside a loop; instead precompute number of divisions and allocate output arrays once.

---

## 7) Time step (`dt`) defaults and guardrails (Revision 2 requirement)

`dt` may be user-specified; if omitted, engine computes a default.

Let:

- `sigma_min = 2 * min(state_table.R)`
- `v_m_max = max(state_table.Fm) / gamma_s`
- safety factor `eta = 0.02` (default)

Default:

- `dt_default = eta * sigma_min / (v_m_max + eps)`

Guardrail warnings:

- compute `dx = v_m_max * dt`
- warn "dt too large" if `dx > 0.1 * sigma_min`
- warn "dt too small" if `dx < 0.002 * sigma_min`

Optional (future): adaptive substepping on spike, but Revision 2 only requires warnings + default.

---

## 8) Initialization (Revision 2 requirement)

### 8.1 Overlap-safe random initialization

Random initialization must **forbid** placing points within `d_min` of one another.

Approach:
- Sequential rejection sampling on sphere:
  - propose point on sphere (possibly anisotropic distribution)
  - reject if within `d_min` of any existing point
- Use a KD-tree during construction for fast proximity queries.
- Provide a maximum attempts safeguard and raise a clear error if packing fraction is too high.

### 8.2 Anisotropy controls (positions and/or headings)

Support configurable anisotropy for initial positions and initial polarity, with a small set of interpretable modes:

- `pos_mode ∈ {uniform, polar_cap, equatorial_band, axial_gradient}`
- `pos_strength ∈ [0,1]` (0=uniform, 1=strongly biased)
- `heading_mode ∈ {isotropic, axial_bias}`
- `heading_strength ∈ [0,1]`

Implementation should be deterministic given RNG seed.

### 8.3 State mixing controls (multi-state)

If multiple `state_id`s are present at init, allow:

- composition: proportions or explicit counts per state
- mixing: `mixing ∈ [0,1]`
  - 1.0 = fully mixed spatially (states independently sampled)
  - 0.0 = maximally segregated (states occupy distinct regions, e.g. caps/bands)

This is an initialization-only control; subsequent mixing emerges from dynamics.

---

## 9) Outputs and data formats (Revision 2 requirement)

The engine should support multiple output targets; the **primary** interoperable output is a **pandas tracks-style DataFrame**.

### 9.1 Primary: pandas tracks DataFrame ("long" format)

Each row is one cell at one time.

Required columns:
- `track_id` (int)
- `t` (float or int)
- `z`, `y`, `x` (float)  — note napari convention uses z,y,x ordering
- `vz`, `vy`, `vx` (float) — instantaneous velocity vector in same coordinate order
- `state_id` (int)

Recommended optional columns:
- any scalar diagnostics per cell (e.g. `contact_count`, local density, `state_var0`, ...)

Notes:
- `track_id` is persistent identity. Division should create a new `track_id` for the new daughter; include `parent_id` optionally.
- Velocity should be recorded consistently (either pre-update or post-update); docstring must state which.

### 9.2 Secondary: lightweight arrays for GUI

For interactive visualization, support a ring-buffer store of:
- `x[t,N,3]`, `p[t,N,3]`, `state_id[t,N]`, optional `state_vars[t,N,M]`

---

## 10) Progress reporting (Revision 2 requirement)

- `engine.run(...)` should optionally display a progress bar (e.g. `tqdm`) when `show_progress=True`.
- Progress must not affect deterministic results (no RNG usage in progress code).

---

## 11) Diagnostics notebook (Revision 2 requirement)

Provide a Jupyter notebook that runs 1–3 simulations and produces diagnostic plots:

- Sphere + points + polarity vectors snapshots
- Constraint drift: max radial error, max |p·n|, max | |p|-1 |
- Mean speed vs time
- Mean contacts vs time
- Candidate pair count vs time (proxy for neighbor settings)
- Optional: min distance statistics on a subsample

Notebook should be easy to point at the installed package or repo.

