# Project Status — Cell Sphere Simulator

_Last updated: YYYY-MM-DD_

---

## Current Version
Backend Revision 2 (in progress)

---

## Current Focus
Phase 3 visualization planning (Phase 2 complete)

---

## Completed Milestones

### Backend Rev2
- [x] cKDTree neighbor search
- [x] Vectorized contact forces + CIL metrics
- [x] dt default heuristic + guardrails
- [x] Overlap-safe initialization
- [x] State mixing + anisotropy controls
- [x] Pandas tracks output
- [x] Diagnostics notebook
- [x] Division performance refactor

---

## Known Issues

### GUI
- [x] Points shading set to spherical
- [x] Surface mesh rendered with a reduced display radius
- [x] Points/surface occlusion improved via mesh opacity and styling
- [x] Consistent visual language between mesh + tracks via shared state coloring

### Backend
- [ ] No profiling results recorded for N=1k–5k
- [ ] Need min-distance diagnostics exposed more clearly
- [ ] Relaxation step could be profiled further

---

## Open Design Questions

- Should 3D visualization remain napari-only?
- Should PyVista be embedded as dock widget?
- Is per-vertex opacity needed for tracks?

---

## Next Milestone

Phase 2 complete. Next: evaluate optional Phase 3 visualization upgrades.

---

## How to Resume Work

1. Read `ARCHITECTURE.md`
2. Check this file
3. Confirm milestone scope before modifying code