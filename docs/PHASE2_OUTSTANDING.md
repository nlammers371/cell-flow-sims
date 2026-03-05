# Phase 2 Outstanding Items (GUI Stabilization)

This note summarizes what remains in **Phase 2 — GUI Stabilization** from `ROADMAP.md` and aligns it with currently tracked GUI issues.

## Outstanding from roadmap

Phase 2 roadmap items have now been implemented:

1. Napari spherical shading
2. Surface radius display fix
3. Density shading improvements
4. Diagnostics improvements
5. Tracks-layer integration for simulated cells

## Clarified deliverable requested

The integration target is now met: backend simulation outputs are consumed as **tracks-layer objects**, and the GUI displays simulated cells through a napari Tracks layer representation.

## Traceability to current status docs

Current GUI issues already indicate incomplete visual consistency and rendering behavior:

- Points shading not set to spherical
- Surface mesh rendered at same radius as physics sphere
- Points/surface occlusion awkward at certain angles
- No consistent visual language between mesh + tracks

These issues have been addressed by the Phase 2 GUI stabilization update.
