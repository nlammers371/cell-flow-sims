from __future__ import annotations

from typing import Iterable
import numpy as np


def _random_tangent(p: np.ndarray, rng: np.random.Generator, eps: float) -> np.ndarray:
    g = rng.normal(size=3)
    u = g - np.dot(g, p) * p
    norm = np.linalg.norm(u)
    if norm <= eps:
        u = np.array([1.0, 0.0, 0.0], dtype=float) - p[0] * p
        norm = np.linalg.norm(u)
    return u / norm


def init_from_napari_tracks(
    df,
    t0: float,
    R_E: float,
    rng: np.random.Generator,
    state_id_default: int = 0,
    state_vars_default: Iterable[float] | None = None,
    eps: float = 1e-12,
):
    """Initialize from a napari tracks DataFrame.

    Expected columns: track_id, t, z, y, x (3D) or track_id, t, y, x (2D).
    Optional: state_id and state_vars_* columns.
    """
    import pandas as pd

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    cols = set(df.columns)
    has_3d = {"z", "y", "x"}.issubset(cols)
    has_2d = {"y", "x"}.issubset(cols) and not has_3d
    if not (has_3d or has_2d):
        raise ValueError("tracks must include x/y (and optionally z)")

    df_t0 = df[df["t"] == t0].copy()
    if df_t0.empty:
        raise ValueError("no tracks at t0")

    if has_3d:
        pos = df_t0[["z", "y", "x"]].to_numpy(dtype=float)
    else:
        pos2 = df_t0[["y", "x"]].to_numpy(dtype=float)
        pos = np.column_stack([np.zeros((pos2.shape[0],)), pos2])

    norms = np.linalg.norm(pos, axis=1)
    norms = np.where(norms > eps, norms, 1.0)
    x = R_E * pos / norms[:, None]

    if "state_id" in cols:
        state_id = df_t0["state_id"].to_numpy(dtype=np.int32)
    else:
        state_id = np.full((x.shape[0],), state_id_default, dtype=np.int32)

    state_var_cols = [c for c in df.columns if c.startswith("state_var_")]
    if state_var_cols:
        state_vars = df_t0[state_var_cols].to_numpy(dtype=float)
    else:
        default_vars = np.asarray(list(state_vars_default or []), dtype=float)
        state_vars = np.tile(default_vars, (x.shape[0], 1))

    # Polarity from finite difference if next frame exists, else random tangent
    p = np.zeros_like(x)
    for idx, row in df_t0.iterrows():
        track_id = row["track_id"]
        df_next = df[(df["track_id"] == track_id) & (df["t"] > t0)].sort_values("t")
        if not df_next.empty:
            nxt = df_next.iloc[0]
            if has_3d:
                nxt_pos = np.array([nxt["z"], nxt["y"], nxt["x"]], dtype=float)
            else:
                nxt_pos = np.array([0.0, nxt["y"], nxt["x"]], dtype=float)
            v = nxt_pos - row[["z", "y", "x"]].to_numpy(dtype=float) if has_3d else nxt_pos - np.array([0.0, row["y"], row["x"]], dtype=float)
            n_i = x[df_t0.index.get_loc(idx)] / R_E
            v_t = v - np.dot(v, n_i) * n_i
            norm_v = np.linalg.norm(v_t)
            if norm_v > eps:
                p[df_t0.index.get_loc(idx)] = v_t / norm_v
            else:
                p[df_t0.index.get_loc(idx)] = _random_tangent(n_i, rng, eps)
        else:
            n_i = x[df_t0.index.get_loc(idx)] / R_E
            p[df_t0.index.get_loc(idx)] = _random_tangent(n_i, rng, eps)

    return x, p, state_id, state_vars
