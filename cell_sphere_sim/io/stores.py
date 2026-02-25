from __future__ import annotations

import numpy as np
import pandas as pd


class TrajectoryStore:
    """Simple in-memory trajectory store for Phase 1."""

    def __init__(self) -> None:
        self.t = []
        self.x = []
        self.p = []
        self.state_id = []
        self.state_vars = []

    def append(
        self,
        t: float,
        x: np.ndarray,
        p: np.ndarray,
        state_id: np.ndarray,
        state_vars: np.ndarray,
        **_: object,
    ) -> None:
        self.t.append(float(t))
        self.x.append(np.array(x, copy=True))
        self.p.append(np.array(p, copy=True))
        self.state_id.append(np.array(state_id, copy=True))
        self.state_vars.append(np.array(state_vars, copy=True))

    def to_dict(self) -> dict:
        return {
            "t": np.asarray(self.t, dtype=float),
            "x": self.x,
            "p": self.p,
            "state_id": self.state_id,
            "state_vars": self.state_vars,
        }


class PandasTracksStore:
    """Pandas tracks-style output store."""

    def __init__(self) -> None:
        self._rows: list[dict[str, object]] = []

    def append(
        self,
        *,
        t: float,
        x: np.ndarray,
        v: np.ndarray,
        state_id: np.ndarray,
        track_id: np.ndarray,
        extra: dict[str, np.ndarray] | None = None,
        **_: object,
    ) -> None:
        if x.shape[0] == 0:
            return
        if extra is None:
            extra = {}
        n = x.shape[0]
        for i in range(n):
            row = {
                "track_id": int(track_id[i]),
                "t": float(t),
                "z": float(x[i, 2]),
                "y": float(x[i, 1]),
                "x": float(x[i, 0]),
                "vz": float(v[i, 2]),
                "vy": float(v[i, 1]),
                "vx": float(v[i, 0]),
                "state_id": int(state_id[i]),
            }
            for key, arr in extra.items():
                row[key] = int(arr[i]) if np.issubdtype(arr.dtype, np.integer) else float(arr[i])
            self._rows.append(row)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._rows)
