# Project Status — Cell Sphere Simulator

_Last updated: YYYY-MM-DD_

---

## Current Version
Backend Revision 2 (in progress)

---

## Current Focus
GUI stabilization (napari-only improvements)

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
- [ ] Points shading not set to spherical
- [ ] Surface mesh rendered at same radius as physics sphere
- [ ] Points/surface occlusion awkward at certain angles
- [ ] No consistent visual language between mesh + tracks

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

Napari-only polish:
1. Enable spherical shading
2. Shrink display sphere radius slightly
3. Clean density shading
4. Re-evaluate aesthetics before considering PyVista

---

## How to Resume Work

1. Read `ARCHITECTURE.md`
2. Check this file
3. Confirm milestone scope before modifying code