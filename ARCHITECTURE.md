# Architecture Overview — Cell Sphere Simulator

This document describes how the system is structured and why.

---

# 1. Backend Engine

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

# 2. Initialization

Overlap-safe rejection sampling using KD-tree.

Supports:
- Position anisotropy modes
- Heading anisotropy modes
- State composition + mixing controls

---

# 3. Outputs

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

# 4. GUI Layer (napari)

Napari renders:
- Tracks layer (primary visual)
- Surface mesh (context + density shading)
- Points layer (optional current position view)

Napari is currently the only 3D renderer.

---

# 5. Performance Expectations

Target scale:
- 1k–3k cells typical
- up to 5k acceptable

Primary bottlenecks:
- Neighbor candidate generation
- Pairwise force vectorization

---

# 6. Future Extensions

- PyVista docked renderer for true sphere glyphs
- PDE reaction–diffusion field solver
- Parameter sweep orchestration framework