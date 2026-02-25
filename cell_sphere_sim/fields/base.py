from __future__ import annotations

from typing import Protocol
import numpy as np


class FieldModel(Protocol):
    C: int

    def sample(self, x: np.ndarray, t: float) -> np.ndarray: ...

    def reset_sources(self) -> None: ...

    def accumulate_sources(self, x: np.ndarray, sources: np.ndarray) -> None: ...

    def step(self, dt: float) -> None: ...


class NullField:
    def __init__(self, C: int = 2) -> None:
        self.C = int(C)

    def sample(self, x: np.ndarray, t: float) -> np.ndarray:
        return np.zeros((len(x), self.C), dtype=x.dtype)

    def reset_sources(self) -> None:
        return None

    def accumulate_sources(self, x: np.ndarray, sources: np.ndarray) -> None:
        return None

    def step(self, dt: float) -> None:
        return None
