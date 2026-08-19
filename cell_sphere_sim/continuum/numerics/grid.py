"""Periodic square FFT grid and differential operators."""

from __future__ import annotations

import numpy as np


class SpectralGrid:
    """Precomputed wave-number arrays for a square periodic domain."""

    def __init__(self, size: int, length: float):
        self.size = int(size)
        self.length = float(length)
        self.dx = self.length / self.size
        axis = np.arange(self.size, dtype=np.float64) * self.dx
        self.x, self.y = np.meshgrid(axis, axis, indexing="xy")

        wave = 2.0 * np.pi * np.fft.fftfreq(self.size, d=self.dx)
        self.kx, self.ky = np.meshgrid(wave, wave, indexing="xy")
        self.k2 = self.kx * self.kx + self.ky * self.ky
        self.k4 = self.k2 * self.k2

        # The 2/3 rule removes all modes at and above N/3 in either direction.
        mode = np.fft.fftfreq(self.size) * self.size
        mx, my = np.meshgrid(mode, mode, indexing="xy")
        cutoff = self.size / 3.0
        self.dealias = (np.abs(mx) < cutoff) & (np.abs(my) < cutoff)
        self.dealias[0, 0] = True

    def fft(self, field: np.ndarray) -> np.ndarray:
        return np.fft.fft2(field)

    def ifft(self, modes: np.ndarray) -> np.ndarray:
        return np.fft.ifft2(modes).real

    def filtered_fft(self, field: np.ndarray) -> np.ndarray:
        return self.fft(field) * self.dealias

    def gradient(self, field: np.ndarray):
        modes = self.fft(field)
        return self.ifft(1j * self.kx * modes), self.ifft(1j * self.ky * modes)

    def laplacian(self, field: np.ndarray) -> np.ndarray:
        return self.ifft(-self.k2 * self.fft(field))

    def divergence(self, field_x: np.ndarray, field_y: np.ndarray) -> np.ndarray:
        return self.ifft(
            1j * self.kx * self.filtered_fft(field_x)
            + 1j * self.ky * self.filtered_fft(field_y)
        )

    def periodic_distance(self, center_x: float, center_y: float):
        dx = np.abs(self.x - center_x)
        dy = np.abs(self.y - center_y)
        dx = np.minimum(dx, self.length - dx)
        dy = np.minimum(dy, self.length - dy)
        return np.sqrt(dx * dx + dy * dy)


def conserved_noise_modes(grid: SpectralGrid, rng, strength: float) -> np.ndarray:
    """Return Fourier modes of the divergence of a white random flux."""

    if strength == 0.0:
        return np.zeros_like(grid.k2, dtype=np.complex128)
    flux_x = rng.standard_normal((grid.size, grid.size))
    flux_y = rng.standard_normal((grid.size, grid.size))
    modes = (
        1j * grid.kx * grid.fft(flux_x)
        + 1j * grid.ky * grid.fft(flux_y)
    ) * grid.dealias
    modes[0, 0] = 0.0
    return float(strength) * modes
