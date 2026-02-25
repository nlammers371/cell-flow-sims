# CHANGELOG_v2.md

## Summary
Revision 2 focuses on backend usability, speed, and safer defaults.

## Performance
- Replaced spatial-hash neighbor generation with SciPy cKDTree query_pairs.
- Vectorized contact force + CIL metric computation over pair index arrays.
- Recommended: allocate division outputs once per step (avoid repeated vstack/concatenate).

## Safety / guardrails
- dt can be omitted; engine computes a default based on max motility and cell diameter.
- Warnings issued for dt that is likely too large (instability) or too small (inefficient).

## Initialization
- Random initialization now enforces no overlaps closer than hard-core d_min.
- Added anisotropy controls for initial positions/headings.
- Added mixing control for multi-state spatial mixing vs segregation at initialization.

## Outputs
- Added primary output store in pandas tracks-style long format:
  - track_id, t, z,y,x, vz,vy,vx, state_id (+ optional extras)
- Division assigns new track_id to daughter; optional parent_id.

## UX / diagnostics
- Added progress bar option for run loops.
- Added backend diagnostics notebook with 1–3 runs and key sanity plots.
- Added backend overview doc explaining modules and class attributes.

