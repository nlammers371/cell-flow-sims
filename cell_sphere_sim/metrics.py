from __future__ import annotations

import numpy as np


def compute_speed(x_prev: np.ndarray, x_next: np.ndarray, dt: float) -> np.ndarray:
    return np.linalg.norm((x_next - x_prev) / dt, axis=1)
