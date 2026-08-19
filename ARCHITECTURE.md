# Architecture Overview — Cell Flow Simulations

This document describes how the system is structured and why.

---

# 1. Spherical Backend Engine

## Core State

Per cell:
- x (N,3) — position (on sphere)
- p (N,3) — polarity (unit, tangent)
- state_id (N,)
- state_vars (N,M)
- paused_until (N,)
- track_id (N,)
- parent_id (N,) optional

---

## Engine Step Order

Each timestep:

1. Sample fields
2. Update state + compute behavior parameters
3. Neighbor candidate generation (cKDTree)
4. Vectorized contact forces + CIL metrics
5. Compute velocity (motility + contact)
6. Explicit Euler position update
7. Project to sphere
8. Parallel transport polarity
9. Apply CIL relaxation + diffusion
10. Handle divisions
11. Optional contact relaxation substeps
12. Update fields
13. Emit diagnostics

---

## Neighbor Finding

SciPy cKDTree:
- query_pairs with radius = sigma_max * (1 + buffer)
- returns index arrays (i_idx, j_idx)

---

## Forces

Vectorized computation:
- Gather pair arrays
- Filter by contact
- Compute repulsion + adhesion
- Scatter-add via np.add.at

---

## Time Step (dt)

- User may supply dt
- Otherwise computed from motility and cell size
- Warnings if dt too large or too small

---

# 2. Periodic Planar Reference Engine

The standalone modules under `cell_sphere_sim/planar/` provide:

- `(N,2)` positions and unit polarities
- rectangular periodic cKDTree neighbor search
- minimum-image pair geometry
- the same contact force magnitude helper as the sphere engine
- planar CIL relaxation and rotational diffusion
- state-specific unrestricted division, motility pauses, and lineage IDs
- event-local periodic hard-disk projection after daughter insertion
- unwrapped positions and collective diagnostics

The planar step orchestration is intentionally separate from the spherical
engine. State tables, division sampling, and the actual force formula are
shared; geometry-specific placement, projection, and transport are not
abstracted into a broad framework.

---

# 3. Continuum Workbench

`cell_sphere_sim/continuum/` is independent of both cell-agent engines:

- `base.py`, `config.py`, and `registry.py` define the model contract and
  declarative controls/presets.
- `numerics/` owns the cached periodic FFT grid, differential operators,
  de-aliasing mask, and conserved flux noise.
- `models/` implements Passive Model B, Active Model B,
  density–polarization MIPS, and Keller–Segel autochemotaxis.
- `engine.py` separates PDE stepping from render cadence and handles retry,
  positivity, deterministic energy descent, and diagnostic history.
- `diagnostics.py`, `export.py`, `comparison.py`, and `sweep.py` are shared by
  the GUI and headless workflows.
- `workbench.py` is a registry-driven Matplotlib client; the models do not
  import Matplotlib.

The continuum and agent paths intentionally share no state semantics: continuum
pixels are coarse-grained fields rather than cells.

---

# 4. Initialization

Overlap-safe rejection sampling using KD-tree.

Supports:
- Position anisotropy modes
- Heading anisotropy modes
- State composition + mixing controls

---

# 5. Outputs

## Primary Output
Pandas tracks-style DataFrame:

Columns:
- track_id
- t
- z, y, x
- vz, vy, vx
- state_id
- optional features

Division creates new track_id.

---

# 6. Visualization

Napari renders:
- Tracks layer (primary visual)
- Surface mesh (context + density shading)
- Points layer (optional current position view)

Napari is currently the only 3D renderer.

`examples/run_2d_sandbox.py` is a separate Matplotlib client of the planar
engine. Its headless mode imports no Matplotlib code.

`examples/run_continuum_workbench.py` is the clearly separated continuum route.
Its controls are generated from `MODEL_REGISTRY`, and its headless mode likewise
imports no Matplotlib code.

---

# 7. Performance Expectations

Target scale:
- 1k–3k cells typical
- up to 5k acceptable

Primary bottlenecks:
- Neighbor candidate generation
- Pairwise force vectorization

---

# 8. Future Extensions

- Planar parameter sweeps and sphere-reference comparisons
- PDE reaction–diffusion field solver
- Optional 3D rendering upgrades
