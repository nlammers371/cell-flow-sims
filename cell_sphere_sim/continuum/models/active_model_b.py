"""Active Model B, with a non-integrable chemical-potential contribution."""

from __future__ import annotations

from ..base import ParameterSpec, PresetSpec
from .model_b import (
    MODEL_B_PARAMETERS,
    ModelB,
    chemical_potential,
    model_b_step,
    passive_free_energy,
)


class ActiveModelB(ModelB):
    key = "active_model_b"
    name = "Active Model B"
    description = "A minimal nonequilibrium extension of Model B."
    equations = (
        "∂tφ = M∇²μ + ∇·Λ",
        "μ = aφ + bφ³ − κ∇²φ + λ|∇φ|²",
    )
    parameter_specs = MODEL_B_PARAMETERS + (
        ParameterSpec(
            "activity", "Activity", 0.5,
            "Nonequilibrium chemical-potential coefficient λ; it is not speed or temperature.",
            group="Activity", symbol="λ", minimum=-3.0, maximum=3.0,
            stability_warning="Large |λ| may require a smaller timestep.",
        ),
    )
    presets = (
        PresetSpec(
            "passive_limit", "Passive limit", "Exactly recovers Passive Model B when λ = 0.",
            {"a": -1.0, "b": 1.0, "kappa": 1.0, "mobility": 1.0, "activity": 0.0},
            "uniform_noise", {"mean": 0.0, "amplitude": 0.03},
        ),
        PresetSpec(
            "active_demix", "Active demixing", "Demixing with broken detailed balance.",
            {"a": -1.0, "b": 1.0, "kappa": 1.0, "mobility": 1.0, "activity": 0.75},
            "uniform_noise", {"mean": 0.0, "amplitude": 0.03},
        ),
        PresetSpec(
            "active_droplet", "Active droplet", "A finite droplet under active chemical-potential dynamics.",
            {"a": -1.0, "b": 1.0, "kappa": 1.0, "mobility": 1.0, "activity": -0.75},
            "droplet", {"mean": -0.75, "inside": 1.0, "radius": 7.0, "interface": 1.2},
        ),
    )

    def step(self, state, grid, dt, parameters, rng, noise_strength):
        return {
            "phi": model_b_step(
                state.fields["phi"], grid, dt, parameters, rng, noise_strength,
                active_lambda=float(parameters["activity"]),
            )
        }

    def derived_fields(self, state, grid, parameters):
        return {
            "mu": chemical_potential(
                state.fields["phi"], grid, parameters, active_lambda=float(parameters["activity"])
            )
        }

    def model_diagnostics(self, state, grid, parameters):
        if float(parameters["activity"]) == 0.0:
            return {"free_energy": passive_free_energy(state.fields["phi"], grid, parameters)}
        # The passive functional is intentionally not reported for λ != 0:
        # active dynamics has no corresponding Lyapunov free energy.
        return {}
