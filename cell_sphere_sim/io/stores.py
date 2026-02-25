from __future__ import annotations

import numpy as np


class TrajectoryStore:
    """Simple in-memory trajectory store for Phase 1."""

    def __init__(self) -> None:
        self.t = []
        self.x = []
        self.p = []
        self.state_id = []
        self.state_vars = []

    def append(self, t: float, x: np.ndarray, p: np.ndarray, state_id: np.ndarray, state_vars: np.ndarray) -> None:
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
