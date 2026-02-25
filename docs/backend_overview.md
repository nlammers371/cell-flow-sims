# Backend overview (Revision 2)

This document summarizes backend module responsibilities, engine attributes, step order, parameters, and outputs.

## Module map

- `cell_sphere_sim/engine.py`: Core simulation loop, time stepping, division, and diagnostics.
- `cell_sphere_sim/forces.py`: Vectorized contact forces and CIL contact metrics.
- `cell_sphere_sim/neighbors.py`: cKDTree neighbor candidate generation.
- `cell_sphere_sim/polarity.py`: Parallel transport, rotational diffusion, and CIL targeting helpers.
- `cell_sphere_sim/division.py`: Division sampling and daughter insertion.
- `cell_sphere_sim/init.py`: Overlap-safe initialization with anisotropy and mixing controls.
- `cell_sphere_sim/io/stores.py`: Output stores (`TrajectoryStore`, `PandasTracksStore`).
- `cell_sphere_sim/state.py`: State table and per-cell behavior lookup.

## SimulationEngine attributes

- `x` (N,3): Positions on the sphere.
- `p` (N,3): Polarity vectors, tangent and unit length.
- `state_id` (N,): Discrete state IDs.
- `state_vars` (N,M): Continuous state variables (M can be 0).
- `track_id` (N,): Persistent identity per cell.
- `parent_id` (N,): Parent track ID for lineage (`-1` if none).
- `v` (N,3): Post-update velocity `(x_new - x_old) / dt`.
- `contact_metrics`: `ContactMetrics` with `contact_count` and `contact_dir_sum`.

## Step order

1) Sample fields and update behavior (`cell_update`).
2) Build neighbor candidates via cKDTree and compute contact forces + CIL metrics.
3) Compute velocity and explicit Euler position update.
4) Project positions to the sphere.
5) Parallel transport polarity, apply CIL relaxation, then diffusion.
6) Optional division (inserts daughters, updates lineage).
7) Optional contact relaxation after division.
8) Accumulate field sources and step field model.
9) Return diagnostics.

## Key parameters

- `R_E`: Sphere radius.
- `gamma_s`: Drag coefficient (overdamped).
- `k_rep`: Repulsion strength.
- `alpha_dmin`: Hard-core ratio. `d_min = alpha_dmin * (R_i + R_j)`.
- `dt`: Time step (may be None; default is computed).
- `neighbor_radius_buffer`: Buffer for cKDTree query radius.
- `record_interval`: Store every N steps.

### dt defaults and warnings

If `dt` is None, the engine computes:

$$
\Delta t = \eta \frac{\sigma_{\min}}{v_{m,\max} + \varepsilon}
$$

with `eta=0.02`, `sigma_min = 2*min(R)`, and `v_m_max = max(Fm)/gamma_s`.
Warnings are emitted if `v_m_max * dt` is too large or too small relative to `sigma_min`.

## Outputs

### Pandas tracks (primary)

`PandasTracksStore` writes a long-format DataFrame with columns:

- `track_id`, `t`, `z`, `y`, `x`, `vz`, `vy`, `vx`, `state_id`
- optional extras (e.g., `parent_id`)

Coordinates are recorded as `z,y,x` to match napari conventions. Velocity is
post-update: `(x_new - x_old) / dt` after projection.

### Lightweight arrays (secondary)

`TrajectoryStore` retains lists of arrays for quick inspection or GUI use.

## Tuning tips

- **Neighbor buffer**: use `neighbor_radius_buffer` in the range 0.05–0.15.
- **dt**: if speeds spike, lower `dt` or increase `alpha_dmin`.
- **Initialization**: prefer `init_random_on_sphere` to avoid large overlaps.
- **Performance**: if candidate pairs explode, reduce buffer or check radii.
