from __future__ import annotations

import numpy as np
import pandas as pd

from cell_sphere_sim.io.tracks_init import init_from_napari_tracks


def make_synthetic_tracks(n_tracks: int, n_steps: int) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(5)
    for tid in range(n_tracks):
        pos = rng.normal(size=3)
        vel = rng.normal(size=3) * 0.1
        for t in range(n_steps):
            pos = pos + vel
            rows.append({"track_id": tid, "t": t, "z": pos[0], "y": pos[1], "x": pos[2]})
    return pd.DataFrame(rows)


def main() -> None:
    rng = np.random.default_rng(7)
    df = make_synthetic_tracks(n_tracks=5, n_steps=4)
    x, p, state_id, state_vars = init_from_napari_tracks(df, t0=0, R_E=10.0, rng=rng)
    print("x shape", x.shape)
    print("p shape", p.shape)
    print("state_id", state_id)
    print("state_vars shape", state_vars.shape)


if __name__ == "__main__":
    main()
