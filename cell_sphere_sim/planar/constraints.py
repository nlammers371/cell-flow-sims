from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .neighbors import (
    candidate_pairs_periodic,
    minimum_image_displacement,
    validate_box_size,
)


@dataclass(frozen=True)
class ProjectionDiagnostics:
    """Audit information for one instantaneous overlap projection."""

    iterations: int
    n_cells_moved: int
    initial_max_overlap: float
    final_max_overlap: float
    max_displacement: float
    rms_displacement: float
    displacement: np.ndarray


def _fallback_normal(i: int, j: int) -> np.ndarray:
    """Return a deterministic direction for exactly coincident centers."""
    phase = ((i + 1) * 0.7548776662466927 + (j + 1) * 0.5698402909980532) % 1.0
    angle = 2.0 * np.pi * phase
    return np.array([np.cos(angle), np.sin(angle)], dtype=float)


def _overlaps(
    x: np.ndarray,
    radii: np.ndarray,
    box: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    query_radius = 2.0 * float(np.max(radii))
    i_idx, j_idx = candidate_pairs_periodic(x, query_radius, box)
    if i_idx.size == 0:
        return i_idx, j_idx, np.empty(0, dtype=float)
    distances = np.linalg.norm(minimum_image_displacement(x[i_idx], x[j_idx], box), axis=1)
    return i_idx, j_idx, radii[i_idx] + radii[j_idx] - distances


def project_seeded_overlaps_periodic(
    x: np.ndarray,
    x_unwrapped: np.ndarray,
    radii: np.ndarray,
    seed_indices: np.ndarray,
    box_size: np.ndarray | tuple[float, float],
    *,
    tolerance: float = 1e-8,
    max_iterations: int = 500,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, ProjectionDiagnostics]:
    """Instantaneously remove overlaps connected to ``seed_indices``.

    Each pair correction is split equally between its cells, preserving the
    unwrapped population center of mass. The active component grows whenever a
    correction reaches another overlapping cell; unrelated pre-existing
    overlap components are left untouched. Pair order alternates between
    passes to reduce deterministic ordering bias.
    """
    box = validate_box_size(box_size)
    wrapped = np.asarray(x, dtype=float)
    unwrapped = np.asarray(x_unwrapped, dtype=float)
    cell_radii = np.asarray(radii, dtype=float)
    seeds = np.asarray(seed_indices, dtype=np.int64)

    if wrapped.ndim != 2 or wrapped.shape[1] != 2:
        raise ValueError("x must have shape (N, 2)")
    if unwrapped.shape != wrapped.shape:
        raise ValueError("x_unwrapped must have the same shape as x")
    if cell_radii.shape != (wrapped.shape[0],):
        raise ValueError("radii must have shape (N,)")
    if not np.all(np.isfinite(wrapped)) or not np.all(np.isfinite(unwrapped)):
        raise ValueError("positions must contain only finite values")
    if not np.all(np.isfinite(cell_radii)) or np.any(cell_radii <= 0.0):
        raise ValueError("radii must be finite and positive")
    seeds_out_of_range = seeds.size and (
        np.any(seeds < 0) or np.any(seeds >= wrapped.shape[0])
    )
    if seeds.ndim != 1 or seeds_out_of_range:
        raise ValueError("seed_indices must contain valid cell indices")
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and non-negative")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    if not np.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be finite and positive")
    if seeds.size == 0:
        diagnostics = ProjectionDiagnostics(
            iterations=0,
            n_cells_moved=0,
            initial_max_overlap=0.0,
            final_max_overlap=0.0,
            max_displacement=0.0,
            rms_displacement=0.0,
            displacement=np.zeros_like(unwrapped),
        )
        return wrapped.copy(), unwrapped.copy(), diagnostics

    wrapped_out = np.mod(wrapped.copy(), box)
    unwrapped_out = unwrapped.copy()
    active = np.zeros(wrapped.shape[0], dtype=bool)
    active[seeds] = True
    total_correction = np.zeros_like(unwrapped_out)
    initial_max_overlap = 0.0
    iterations = 0

    for pass_index in range(max_iterations):
        i_idx, j_idx, overlap = _overlaps(wrapped_out, cell_radii, box)
        selected = np.flatnonzero(
            (overlap > tolerance) & (active[i_idx] | active[j_idx])
        )
        if selected.size == 0:
            break
        if pass_index == 0:
            initial_max_overlap = float(np.max(overlap[selected]))
        if pass_index % 2:
            selected = selected[::-1]

        corrected_any = False
        for pair_index in selected:
            i = int(i_idx[pair_index])
            j = int(j_idx[pair_index])
            if not (active[i] or active[j]):
                continue
            dvec = minimum_image_displacement(wrapped_out[i], wrapped_out[j], box)
            distance = float(np.linalg.norm(dvec))
            penetration = float(cell_radii[i] + cell_radii[j] - distance)
            if penetration <= tolerance:
                continue
            normal = dvec / distance if distance > eps else _fallback_normal(i, j)
            correction = 0.5 * penetration * normal
            unwrapped_out[i] += correction
            unwrapped_out[j] -= correction
            total_correction[i] += correction
            total_correction[j] -= correction
            wrapped_out[i] = np.mod(unwrapped_out[i], box)
            wrapped_out[j] = np.mod(unwrapped_out[j], box)
            active[i] = True
            active[j] = True
            corrected_any = True

        if not corrected_any:
            break
        iterations = pass_index + 1

    i_idx, j_idx, overlap = _overlaps(wrapped_out, cell_radii, box)
    active_overlap = overlap[active[i_idx] | active[j_idx]]
    final_max_overlap = (
        max(0.0, float(np.max(active_overlap))) if active_overlap.size else 0.0
    )
    if final_max_overlap > tolerance:
        raise RuntimeError(
            "division overlap projection did not converge: "
            f"residual={final_max_overlap:.6g}, tolerance={tolerance:.6g}, "
            f"iterations={max_iterations}"
        )

    displacement_norm = np.linalg.norm(total_correction, axis=1)
    moved = displacement_norm > eps
    diagnostics = ProjectionDiagnostics(
        iterations=iterations,
        n_cells_moved=int(np.count_nonzero(moved)),
        initial_max_overlap=initial_max_overlap,
        final_max_overlap=final_max_overlap,
        max_displacement=float(np.max(displacement_norm)) if displacement_norm.size else 0.0,
        rms_displacement=(
            float(np.sqrt(np.mean(displacement_norm[moved] ** 2))) if np.any(moved) else 0.0
        ),
        displacement=total_correction.copy(),
    )
    return wrapped_out, unwrapped_out, diagnostics
