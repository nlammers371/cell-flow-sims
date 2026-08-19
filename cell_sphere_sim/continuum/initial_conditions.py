"""Initial conditions shared by the continuum models."""

from __future__ import annotations

import numpy as np


def _restore_mean(field: np.ndarray, target: float) -> np.ndarray:
    return field + (float(target) - float(np.mean(field)))


def uniform_noise(grid, rng, mean: float, amplitude: float) -> np.ndarray:
    field = float(mean) + float(amplitude) * rng.standard_normal((grid.size, grid.size))
    return _restore_mean(field, mean)


def smooth_droplet(grid, mean: float, radius: float, inside: float, interface: float = 1.0):
    distance = grid.periodic_distance(0.5 * grid.length, 0.5 * grid.length)
    shape = 0.5 * (1.0 - np.tanh((distance - radius) / max(interface, grid.dx)))
    field = float(mean) + (float(inside) - float(mean)) * shape
    return _restore_mean(field, mean)


def multiple_droplets(
    grid, rng, mean: float, radius: float, inside: float, count: int, interface: float = 1.0
):
    field = np.full((grid.size, grid.size), float(mean), dtype=np.float64)
    for _ in range(int(count)):
        cx, cy = rng.uniform(0.0, grid.length, size=2)
        distance = grid.periodic_distance(cx, cy)
        shape = 0.5 * (1.0 - np.tanh((distance - radius) / max(interface, grid.dx)))
        field += (float(inside) - float(mean)) * shape
    return _restore_mean(field, mean)


def stripe_interface(grid, mean: float, amplitude: float, interface: float = 1.0):
    # A cosine level set makes the two required periodic interfaces explicit.
    field = float(mean) + float(amplitude) * np.tanh(
        np.cos(2.0 * np.pi * grid.x / grid.length) * grid.length / max(interface, grid.dx)
    )
    return _restore_mean(field, mean)


def radial_profile(grid, mean: float, amplitude: float):
    distance = grid.periodic_distance(0.5 * grid.length, 0.5 * grid.length)
    field = float(mean) + float(amplitude) * np.cos(4.0 * np.pi * distance / grid.length)
    return _restore_mean(field, mean)


def make_scalar_initial_condition(
    grid,
    rng,
    kind: str,
    mean: float,
    amplitude: float,
    radius: float,
    count: int,
    inside: float,
    interface: float = 1.0,
):
    if kind == "uniform_noise":
        return uniform_noise(grid, rng, mean, amplitude)
    if kind == "droplet":
        return smooth_droplet(grid, mean, radius, inside, interface)
    if kind == "multiple_droplets":
        return multiple_droplets(grid, rng, mean, radius, inside, count, interface)
    if kind == "single_interface":
        return stripe_interface(grid, mean, amplitude, interface)
    if kind == "radial":
        return radial_profile(grid, mean, amplitude)
    raise ValueError(f"Unknown initial condition {kind!r}")
