from __future__ import annotations

import numpy as np

from ..division import sample_divisions
from ..state import BehaviorParams
from .neighbors import validate_box_size


def apply_planar_divisions(
    x: np.ndarray,
    x_unwrapped: np.ndarray,
    p: np.ndarray,
    state_id: np.ndarray,
    state_vars: np.ndarray,
    paused_until: np.ndarray,
    track_id: np.ndarray,
    parent_id: np.ndarray,
    next_track_id: int,
    t: float,
    behavior: BehaviorParams,
    box_size: np.ndarray | tuple[float, float],
    division_separation_factor: float,
    rng: np.random.Generator,
    dt: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    int,
    np.ndarray,
]:
    """Replace dividing planar parents with two symmetric daughters.

    Both daughters retain the parent's radius through exact state inheritance.
    Their center-to-center birth separation is
    ``division_separation_factor * 2 * R_parent``. Both receive new track IDs;
    the original parent track terminates at division.
    """
    box = validate_box_size(box_size)
    divide_mask = sample_divisions(behavior.lambda_div, dt=dt, rng=rng)
    div_idx = np.flatnonzero(divide_mask)
    if div_idx.size == 0:
        return (
            x,
            x_unwrapped,
            p,
            state_id,
            state_vars,
            paused_until,
            track_id,
            parent_id,
            next_track_id,
            div_idx,
        )

    n_cells = x.shape[0]
    n_divisions = int(div_idx.size)
    out_n = n_cells + n_divisions

    x_unwrapped_out = np.empty((out_n, 2), dtype=x_unwrapped.dtype)
    p_out = np.empty((out_n, 2), dtype=p.dtype)
    state_id_out = np.empty((out_n,), dtype=state_id.dtype)
    state_vars_out = np.empty((out_n, state_vars.shape[1]), dtype=state_vars.dtype)
    paused_out = np.empty((out_n,), dtype=paused_until.dtype)
    track_id_out = np.empty((out_n,), dtype=track_id.dtype)
    parent_id_out = np.empty((out_n,), dtype=parent_id.dtype)

    x_unwrapped_out[:n_cells] = x_unwrapped
    p_out[:n_cells] = p
    state_id_out[:n_cells] = state_id
    state_vars_out[:n_cells] = state_vars
    paused_out[:n_cells] = paused_until
    track_id_out[:n_cells] = track_id
    parent_id_out[:n_cells] = parent_id

    angles = rng.uniform(0.0, 2.0 * np.pi, size=n_divisions)
    axes = np.column_stack((np.cos(angles), np.sin(angles)))
    # Half of the requested daughter center-to-center distance.
    offsets = division_separation_factor * behavior.R[div_idx, None] * axes
    first_positions = x_unwrapped[div_idx] + offsets
    second_positions = x_unwrapped[div_idx] - offsets
    appended = np.arange(n_cells, out_n)

    x_unwrapped_out[div_idx] = first_positions
    x_unwrapped_out[appended] = second_positions
    p_out[div_idx] = p[div_idx]
    p_out[appended] = p[div_idx]
    state_id_out[appended] = state_id[div_idx]
    state_vars_out[appended] = state_vars[div_idx]

    # The event is applied after the current mechanics step. Starting the
    # pause at t+dt therefore provides the full state-specific pause duration.
    pause_end = np.maximum(paused_until[div_idx], t + dt + behavior.tau_div[div_idx])
    paused_out[div_idx] = pause_end
    paused_out[appended] = pause_end

    old_track_id = track_id[div_idx]
    daughter_ids = next_track_id + np.arange(2 * n_divisions, dtype=track_id.dtype)
    track_id_out[div_idx] = daughter_ids[0::2]
    track_id_out[appended] = daughter_ids[1::2]
    parent_id_out[div_idx] = old_track_id
    parent_id_out[appended] = old_track_id

    x_out = np.mod(x_unwrapped_out, box)
    return (
        x_out,
        x_unwrapped_out,
        p_out,
        state_id_out,
        state_vars_out,
        paused_out,
        track_id_out,
        parent_id_out,
        next_track_id + 2 * n_divisions,
        div_idx,
    )
