from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp


@dataclass(frozen=True)
class FreeMotionEMFit:
    """Maximum-likelihood fit of the discretized hidden-angle model."""

    speed: float
    rotational_diffusion: float
    step_sigma: float
    log_likelihood: float
    n_iterations: int
    converged: bool
    n_segments: int
    n_steps: int
    n_angle_bins: int
    log_likelihood_history: tuple[float, ...]


def _validate_segments(segments: list[np.ndarray]) -> list[np.ndarray]:
    clean: list[np.ndarray] = []
    for segment in segments:
        values = np.asarray(segment, dtype=float)
        if values.ndim != 2 or values.shape[1] != 2:
            raise ValueError("each segment must have shape (n_steps, 2)")
        if values.shape[0] == 0:
            continue
        if not np.all(np.isfinite(values)):
            raise ValueError("segments must contain only finite values")
        clean.append(values)
    if not clean:
        raise ValueError("at least one non-empty segment is required")
    return clean


def _wrapped_transition(
    rotational_diffusion: float,
    dt: float,
    angles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return transition probabilities and logs on a circular angle grid."""
    variance = 2.0 * rotational_diffusion * dt
    delta = angles[None, :] - angles[:, None]
    images = delta[..., None] + 2.0 * np.pi * np.arange(-4, 5)
    log_density = logsumexp(-(images**2) / (2.0 * variance), axis=-1)
    log_transition = log_density - logsumexp(log_density, axis=1, keepdims=True)
    return np.exp(log_transition), log_transition


def _emission_log_probability(
    displacement: np.ndarray,
    speed: float,
    step_sigma: float,
    dt: float,
    headings: np.ndarray,
) -> np.ndarray:
    means = speed * dt * headings
    residual = displacement[:, None, :] - means[None, :, :]
    squared_error = np.sum(residual**2, axis=2)
    variance = step_sigma**2
    return -np.log(2.0 * np.pi * variance) - squared_error / (2.0 * variance)


def _forward_backward(
    log_emission: np.ndarray,
    transition: np.ndarray,
    log_transition: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    n_steps, n_angles = log_emission.shape
    log_alpha = np.empty_like(log_emission)
    log_alpha[0] = -np.log(n_angles) + log_emission[0]
    for step in range(1, n_steps):
        log_alpha[step] = log_emission[step] + logsumexp(
            log_alpha[step - 1, :, None] + log_transition,
            axis=0,
        )
    log_likelihood = float(logsumexp(log_alpha[-1]))

    log_beta = np.zeros_like(log_emission)
    for step in range(n_steps - 2, -1, -1):
        log_beta[step] = logsumexp(
            log_transition
            + log_emission[step + 1, None, :]
            + log_beta[step + 1, None, :],
            axis=1,
        )

    gamma = np.exp(log_alpha + log_beta - log_likelihood)
    gamma /= np.sum(gamma, axis=1, keepdims=True)

    transition_counts = np.zeros_like(transition)
    for step in range(n_steps - 1):
        log_xi = (
            log_alpha[step, :, None]
            + log_transition
            + log_emission[step + 1, None, :]
            + log_beta[step + 1, None, :]
            - log_likelihood
        )
        xi = np.exp(log_xi)
        xi /= np.sum(xi)
        transition_counts += xi
    return log_likelihood, gamma, transition_counts


def _initial_parameters(segments: list[np.ndarray], dt: float) -> tuple[float, float, float]:
    displacement = np.concatenate(segments, axis=0)
    lengths = np.linalg.norm(displacement, axis=1)
    speed = max(float(np.median(lengths) / dt), 1e-8)

    direction_correlations: list[np.ndarray] = []
    for segment in segments:
        norms = np.linalg.norm(segment, axis=1)
        if segment.shape[0] < 2:
            continue
        unit = np.divide(
            segment,
            norms[:, None],
            out=np.zeros_like(segment),
            where=norms[:, None] > 1e-12,
        )
        valid = (norms[:-1] > 1e-12) & (norms[1:] > 1e-12)
        if np.any(valid):
            direction_correlations.append(np.sum(unit[:-1][valid] * unit[1:][valid], axis=1))
    if direction_correlations:
        mean_correlation = float(np.mean(np.concatenate(direction_correlations)))
        mean_correlation = float(np.clip(mean_correlation, 1e-4, 0.9999))
        rotational_diffusion = -np.log(mean_correlation) / dt
    else:
        rotational_diffusion = 1.0
    rotational_diffusion = float(np.clip(rotational_diffusion, 1e-4, 50.0))

    radial_residual = lengths - speed * dt
    step_sigma = max(float(np.std(radial_residual)), 0.1 * float(np.median(lengths)), 1e-6)
    return speed, rotational_diffusion, step_sigma


def fit_free_motion_em(
    segments: list[np.ndarray],
    *,
    dt: float,
    n_angle_bins: int = 72,
    max_iterations: int = 50,
    tolerance: float = 1e-5,
    initial_speed: float | None = None,
    initial_rotational_diffusion: float | None = None,
    initial_step_sigma: float | None = None,
    fixed_step_sigma: float | None = None,
    fixed_rotational_diffusion: float | None = None,
) -> FreeMotionEMFit:
    """Fit speed, rotational diffusion, and isotropic step noise by EM.

    The only latent state is a discretized polarity angle. Each observed
    displacement is Gaussian around ``speed * dt * heading(theta)``. Successive
    headings follow wrapped angular diffusion with variance ``2 * Dr * dt``.

    ``fixed_step_sigma`` supports plug-in conditional fits for known
    subpopulations after estimating the observation/model-discrepancy scale on
    the global population.
    """
    clean = _validate_segments(segments)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")
    if n_angle_bins < 12:
        raise ValueError("n_angle_bins must be at least 12")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")

    default_speed, default_dr, default_sigma = _initial_parameters(clean, dt)
    speed = default_speed if initial_speed is None else float(initial_speed)
    rotational_diffusion = (
        default_dr
        if initial_rotational_diffusion is None
        else float(initial_rotational_diffusion)
    )
    step_sigma = default_sigma if initial_step_sigma is None else float(initial_step_sigma)
    if fixed_step_sigma is not None:
        step_sigma = float(fixed_step_sigma)
    if fixed_rotational_diffusion is not None:
        rotational_diffusion = float(fixed_rotational_diffusion)
    if speed <= 0.0 or rotational_diffusion <= 0.0 or step_sigma <= 0.0:
        raise ValueError("initial and fixed parameters must be positive")

    angles = 2.0 * np.pi * np.arange(n_angle_bins) / n_angle_bins
    headings = np.column_stack((np.cos(angles), np.sin(angles)))
    n_steps = int(sum(segment.shape[0] for segment in clean))
    history: list[float] = []
    converged = False

    for iteration in range(max_iterations):
        transition, log_transition = _wrapped_transition(rotational_diffusion, dt, angles)
        total_log_likelihood = 0.0
        expected_heading_dot_displacement = 0.0
        expected_squared_error = 0.0
        transition_counts = np.zeros_like(transition)

        segment_posteriors: list[np.ndarray] = []
        for segment in clean:
            log_emission = _emission_log_probability(
                segment, speed, step_sigma, dt, headings
            )
            log_likelihood, gamma, counts = _forward_backward(
                log_emission, transition, log_transition
            )
            total_log_likelihood += log_likelihood
            transition_counts += counts
            segment_posteriors.append(gamma)

        history.append(total_log_likelihood)
        for segment, gamma in zip(clean, segment_posteriors):
            expected_heading = gamma @ headings
            expected_heading_dot_displacement += float(
                np.sum(segment * expected_heading)
            )

        speed = max(expected_heading_dot_displacement / (n_steps * dt), 1e-10)

        if fixed_step_sigma is None:
            for segment, gamma in zip(clean, segment_posteriors):
                expected_heading = gamma @ headings
                expected_squared_error += float(
                    np.sum(segment**2)
                    - 2.0 * speed * dt * np.sum(segment * expected_heading)
                    + segment.shape[0] * (speed * dt) ** 2
                )
            step_sigma = np.sqrt(max(expected_squared_error / (2.0 * n_steps), 1e-16))

        if fixed_rotational_diffusion is None and np.sum(transition_counts) > 0.0:
            def negative_expected_transition_log_likelihood(candidate: float) -> float:
                _, candidate_log_transition = _wrapped_transition(candidate, dt, angles)
                return -float(np.sum(transition_counts * candidate_log_transition))

            optimum = minimize_scalar(
                negative_expected_transition_log_likelihood,
                method="bounded",
                bounds=(1e-4, 50.0),
                options={"xatol": 1e-6},
            )
            rotational_diffusion = float(optimum.x)

        if len(history) >= 2:
            improvement = history[-1] - history[-2]
            scale = max(1.0, abs(history[-2]))
            if abs(improvement) <= tolerance * scale:
                converged = True
                break

    transition, log_transition = _wrapped_transition(rotational_diffusion, dt, angles)
    final_log_likelihood = 0.0
    for segment in clean:
        log_emission = _emission_log_probability(segment, speed, step_sigma, dt, headings)
        log_likelihood, _, _ = _forward_backward(
            log_emission, transition, log_transition
        )
        final_log_likelihood += log_likelihood

    return FreeMotionEMFit(
        speed=speed,
        rotational_diffusion=rotational_diffusion,
        step_sigma=step_sigma,
        log_likelihood=final_log_likelihood,
        n_iterations=iteration + 1,
        converged=converged,
        n_segments=len(clean),
        n_steps=n_steps,
        n_angle_bins=n_angle_bins,
        log_likelihood_history=tuple(history),
    )
