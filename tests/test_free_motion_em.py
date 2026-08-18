import numpy as np

from cell_sphere_sim.inference import fit_free_motion_em


def _synthetic_segments(
    *,
    speed: float,
    rotational_diffusion: float,
    step_sigma: float,
    dt: float,
    seed: int,
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    segments = []
    for _ in range(80):
        n_steps = 24
        theta = np.empty(n_steps)
        theta[0] = rng.uniform(0.0, 2.0 * np.pi)
        theta[1:] = theta[0] + np.cumsum(
            rng.normal(scale=np.sqrt(2.0 * rotational_diffusion * dt), size=n_steps - 1)
        )
        heading = np.column_stack((np.cos(theta), np.sin(theta)))
        displacement = speed * dt * heading + rng.normal(
            scale=step_sigma, size=(n_steps, 2)
        )
        segments.append(displacement)
    return segments


def test_free_motion_em_recovers_synthetic_parameters():
    dt = 0.05
    segments = _synthetic_segments(
        speed=12.0,
        rotational_diffusion=0.8,
        step_sigma=0.12,
        dt=dt,
        seed=7,
    )
    fit = fit_free_motion_em(
        segments,
        dt=dt,
        n_angle_bins=72,
        max_iterations=40,
        tolerance=1e-6,
    )

    assert fit.converged
    assert np.isclose(fit.speed, 12.0, rtol=0.08)
    assert np.isclose(fit.rotational_diffusion, 0.8, rtol=0.25)
    assert np.isclose(fit.step_sigma, 0.12, rtol=0.12)


def test_free_motion_em_can_hold_global_noise_fixed_for_subgroup_fit():
    dt = 0.05
    segments = _synthetic_segments(
        speed=8.0,
        rotational_diffusion=0.5,
        step_sigma=0.15,
        dt=dt,
        seed=11,
    )
    fit = fit_free_motion_em(
        segments,
        dt=dt,
        fixed_step_sigma=0.15,
        n_angle_bins=72,
        max_iterations=40,
    )

    assert fit.step_sigma == 0.15
    assert np.isclose(fit.speed, 8.0, rtol=0.1)
    assert np.isclose(fit.rotational_diffusion, 0.5, rtol=0.3)
