# Project Status — Cell Flow Simulations

_Last updated: 2026-08-18_

## Current version

Backend Revision 2 with periodic planar MVP and continuum workbench.

## Current focus

Use the standalone 2D sandbox to test, tune, and visualize the preserved cell
mechanics before expanding spherical visualization or adding new biology.

## Completed milestones

### Spherical backend Revision 2

- [x] cKDTree neighbor search
- [x] Vectorized contact forces and CIL metrics
- [x] Timestep defaults and guardrails
- [x] Sphere initialization, state mixing, and anisotropy controls
- [x] Division and persistent lineage tracking
- [x] Trajectory and Pandas track outputs
- [x] Napari sphere viewer stabilization

### Periodic planar MVP

- [x] Deterministic rectangular periodic initialization
- [x] Separate conservative initial-clearance control
- [x] Periodic cKDTree neighbors and minimum-image contact geometry
- [x] Force-law parity with the spherical model
- [x] Overdamped motion, planar CIL, and rotational diffusion
- [x] Unwrapped trajectories and collective diagnostics
- [x] Interactive and headless Matplotlib sandbox
- [x] State-specific unrestricted proliferation and lineage tracking
- [x] Auditable event-local daughter overlap projection
- [x] Focused planar regression and integration tests
- [x] Backend, sandbox, and Napari dependency separation

### Continuum workbench

- [x] Shared float64 periodic FFT core with 2/3 de-aliasing
- [x] Passive and Active Model B with exact passive-limit parity
- [x] Density–polarization MIPS and Keller–Segel autochemotaxis
- [x] Dynamic and quasistatic signal modes
- [x] Registry-driven interactive and headless route
- [x] Mass, structure, cluster, energy, and model-specific diagnostics
- [x] Adaptive invalid-step rejection without material density clipping
- [x] Reproducible arrays, diagnostics, image, metadata, and animation export
- [x] Shared-initial-field comparison and operational-map sweeps
- [x] Saved per-model regression trajectories and focused acceptance tests
- [x] Auditable equations, scaling, assumptions, and limitations guide

## Known limitations

### Planar backend

- [ ] Coupling the discrete cell agents to PDE fields is not implemented.
- [ ] Proliferation has no cell-cycle stages, crowding regulation, or death.
- [ ] Division projection is instantaneous; finite-duration cytokinesis is deferred.
- [ ] The existing vector CIL relaxation has an exact-antipode edge case.
- [ ] The Pandas track store is still 3D-specific; `TrajectoryStore` is
  dimension-agnostic.
- [ ] Parameter sweep and result aggregation tooling is not yet available.

### Spherical backend and GUI

- [ ] No profiling results are recorded for `N=1k–5k`.
- [ ] Division contact relaxation could be profiled further.
- [ ] Napari remains the only 3D renderer.

## Next milestone

Exercise collective regimes in the 2D sandbox, establish reference parameter
sets and diagnostics, and use them to validate comparable low-curvature sphere
runs. Optional 3D rendering upgrades remain deferred.

## How to resume work

1. Read `ARCHITECTURE.md` and `docs/PLANAR_2D.md`.
2. Run the backend tests and headless sandbox smoke test.
3. Start interactive exploration from `configs/sim_2d.yaml`.
4. Record reference parameter sets before changing model assumptions.
