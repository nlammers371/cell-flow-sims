# Periodic planar model

The planar engine is a focused reference and exploration path for the existing
cell-mechanics model. It changes geometry and boundaries, not the substantive
motility, contact, adhesion, CIL, noise, or state assumptions.

## State and coordinates

For `N` cells, positions and unit polarities have shapes `(N, 2)`:

```text
x_i = (x_i, y_i),       p_i = (p_xi, p_yi),       ||p_i|| = 1
```

The rectangular box vector is `L = (Lx, Ly)`. Current positions are represented
inside `[0, Lx) x [0, Ly)`. For a candidate pair, the displacement from cell
`j` to cell `i` is

```text
dvec = x_i - x_j
dvec = dvec - L * round(dvec / L)
d = ||dvec||
```

This minimum-image displacement is used consistently for contact detection,
forces, CIL directions, and diagnostics. Periodic candidate pairs come from
`scipy.spatial.cKDTree` with `boxsize=L`.

## Contact mechanics

For state-dependent radii `R_i` and `R_j`, define

```text
sigma = R_i + R_j
d_min = alpha_dmin * sigma
d_eff = max(d, d_min + eps)
```

A pair contacts when `d < sigma`. The force magnitudes are unchanged from the
sphere engine:

```text
F_rep = k_rep * ((sigma - d_eff) / (d_eff - d_min))^(3/2)
delta = sigma - d_eff
r_bar = (R_i + R_j) / 2
F_adh = (w_i * w_j / r_bar) * delta
F_mag = F_rep - F_adh
```

Cell `i` receives `F_mag * dvec / d_eff`; cell `j` receives its negative. The
geometry-independent magnitude calculation is shared with the spherical engine
to prevent silent formula drift.

For CIL metrics, cell `i` accumulates the direction toward `j`, `-dvec/d`, and
cell `j` accumulates `dvec/d`. Contact counts are accumulated for both cells.

## Motion and polarity

The overdamped velocity and explicit Euler update are

```text
v_i = (Fm_i * p_i + F_contact_i) / gamma_s
x_unwrapped_i <- x_unwrapped_i + dt * v_i
x_i <- (x_i + dt * v_i) mod L
```

The planar engine preserves the existing polarity rule in this order:

1. Normalize the negative CIL contact-direction sum to obtain a flee target.
2. Relax with `p <- p_flee + exp(-fcil * dt) * (p - p_flee)`.
3. Draw `delta ~ Normal(0, sqrt(2 * Dr * dt))`.
4. Rotate the planar vector by `delta` and normalize it.

There is no parallel transport in a plane. The current vector relaxation has a
narrow degeneracy if the polarity and flee target are exact antipodes at the
specific relaxation value that makes their weighted sum zero. This limitation
is intentionally retained rather than replacing CIL with a new angular law.

## Initialization clearance

`init_random_periodic` uses deterministic sequential rejection sampling with
uniform box proposals and isotropic polarity angles. Heterogeneous pairs must
satisfy

```text
d_periodic >= initial_min_separation_factor * (R_i + R_j)
```

The default factor is `0.9`. It is an initial-condition control only.
`alpha_dmin`, by contrast, sets the force-law hard-core regularization on every
step. Keeping them separate avoids placing the default population close to the
large-force region around `d_min` without altering the interaction law.

Placement uses periodic distances across box edges. If rejection sampling
exhausts `max_attempts_per_cell`, it raises a packing-density error with the
parameters that can be relaxed.

## Diagnostics

Each `step` returns:

- `polarization`: magnitude of the population mean polarity.
- `nematic_order`: magnitude of `mean(exp(2 i theta))`.
- `mean_speed`: mean of `||v_i||`.
- `mean_contacts`: mean per-cell contact count.
- `n_contact_pairs`: number of undirected pairs with `d < R_i + R_j`.
- `min_d_contact`: minimum distance among contacting pairs, or NaN if none.
- `largest_cluster_fraction`: largest connected contact component divided by
  total cell count; isolated cells are components of size one.
- `mean_squared_displacement`: mean squared displacement from initial unwrapped
  positions.
- `n_candidates` and `n_cells`: neighbor-search and population diagnostics.

These are geometric and kinematic summaries; no biological interpretation is
assigned to them.

## API and output

The public engine surface parallels the sphere engine:

- `PlanarSimulationEngine`, `PlanarParams`, `step(t)`, and `run(...)`.
- Public `x`, `x_unwrapped`, `p`, `v`, `state_id`, `state_vars`, `track_id`, and
  `contact_metrics` arrays.
- Injected `numpy.random.Generator` and a `cell_update` state/behavior hook.

`TrajectoryStore` accepts the `(N, 2)` arrays without changes. The existing
`PandasTracksStore` remains 3D-specific so its Napari `(z, y, x)` contract is not
broken.

## Unsupported features and validation role

The first planar MVP does not implement cell division, lineage events, or PDE
field coupling. Passing `division_enabled=True` raises `NotImplementedError`.
It also adds no alignment, velocity matching, or new torque.

The planar path is intended to become a simple reference for tuning and for
checking spherical runs in regimes where curvature is weak. Direct force parity
tests already ensure both geometries use the same pair magnitude law.
