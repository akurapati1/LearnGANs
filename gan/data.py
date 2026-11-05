import numpy as np


def ring_modes(n_modes: int = 8, radius: float = 2.0) -> np.ndarray:
    ang = 2 * np.pi * np.arange(n_modes) / n_modes
    return radius * np.stack([np.cos(ang), np.sin(ang)], 1)


def sample_real(n: int, rng: np.random.Generator, n_modes: int = 8, std: float = 0.1) -> np.ndarray:
    centers = ring_modes(n_modes)
    return centers[rng.integers(0, n_modes, n)] + rng.normal(0, std, (n, 2))
