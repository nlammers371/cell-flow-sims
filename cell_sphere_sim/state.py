from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple
import numpy as np


@dataclass
class StateTable:
    R: np.ndarray
    Fm: np.ndarray
    Dr: np.ndarray
    fcil: np.ndarray
    w: np.ndarray
    lambda_div: np.ndarray
    tau_div: np.ndarray


class BehaviorParams(NamedTuple):
    R: np.ndarray
    Fm: np.ndarray
    Dr: np.ndarray
    fcil: np.ndarray
    w: np.ndarray
    lambda_div: np.ndarray
    tau_div: np.ndarray


def lookup_behavior(state_id: np.ndarray, state_table: StateTable) -> BehaviorParams:
    """Lookup per-cell behavior parameters from a StateTable.

    This is a pure table lookup with no modulation; callers can layer
    custom behavior logic on top.
    """
    return BehaviorParams(
        R=state_table.R[state_id],
        Fm=state_table.Fm[state_id],
        Dr=state_table.Dr[state_id],
        fcil=state_table.fcil[state_id],
        w=state_table.w[state_id],
        lambda_div=state_table.lambda_div[state_id],
        tau_div=state_table.tau_div[state_id],
    )
