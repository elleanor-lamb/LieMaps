import numpy as np
from dataclasses import dataclass
from typing import Callable


@dataclass
class SymplecticMap2D:
    """Defines a 2D symplectic map (z, delta) -> (z', delta')"""
    map_func: Callable[[float, float], tuple[float, float]]

    def apply(self, z: float, delta: float) -> tuple[float, float]:
        return self.map_func(z, delta)


def rf_drift(eta: float) -> SymplecticMap2D:
    """
    Returns a symplectic RF drift map:
        z' = z + eta * delta
        delta' = delta
    """

    def drift_map(z, delta):
        return z + eta * delta, delta

    return SymplecticMap2D(map_func=drift_map)


def rf_kick(V: float, h: float, phi_s: float, beta: float, E0: float, q: float = 1.0) -> SymplecticMap2D:
    """
    Returns a symplectic RF kick map:
        delta' = delta + (q * V / (beta^2 * E0)) * sin(h*z + phi_s)
        z' = z
    """
    kick_strength = q * V / (beta ** 2 * E0)

    def kick_map(z, delta):
        return z, delta + kick_strength * np.sin(h * z + phi_s)

    return SymplecticMap2D(map_func=kick_map)