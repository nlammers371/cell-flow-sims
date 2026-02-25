from __future__ import annotations

from typing import Any
import numpy as np

from .state import StateTable
from .engine import SimParams


def _as_array(data: Any, dtype=float) -> np.ndarray:
    return np.asarray(data, dtype=dtype)


def state_table_from_dict(cfg: dict[str, Any]) -> StateTable:
    return StateTable(
        R=_as_array(cfg["R"]),
        Fm=_as_array(cfg["Fm"]),
        Dr=_as_array(cfg["Dr"]),
        fcil=_as_array(cfg["fcil"]),
        w=_as_array(cfg["w"]),
        lambda_div=_as_array(cfg["lambda_div"]),
        tau_div=_as_array(cfg["tau_div"]),
    )


def sim_params_from_dict(cfg: dict[str, Any]) -> SimParams:
    return SimParams(
        R_E=float(cfg["R_E"]),
        gamma_s=float(cfg["gamma_s"]),
        k_rep=float(cfg["k_rep"]),
        alpha_dmin=float(cfg["alpha_dmin"]),
        eps=float(cfg["eps"]),
        dt=float(cfg["dt"]),
        neighbor_cell_size=float(cfg["neighbor_cell_size"]),
        record_interval=int(cfg["record_interval"]),
        division_enabled=bool(cfg.get("division_enabled", True)),
        split_scale=float(cfg.get("split_scale", 0.5)),
        relax_substeps=int(cfg.get("relax_substeps", 5)),
        relax_dt_scale=float(cfg.get("relax_dt_scale", 0.2)),
    )
