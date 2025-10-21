import numpy as np

def rf_drift(eta):
    """
    Returns a symplectic drift map function for RF system.
    This updates longitudinal position z using current delta.
    """
    def drift(z, delta):
        z_new = z + eta * delta
        return z_new, delta
    return drift


def rf_kick(V, h, phi_s, beta, E0):
    """
    Returns a symplectic RF kick map function.
    This updates delta using current position z.
    """
    def kick(z, delta):
        delta_new = delta + (V / (beta**2 * E0)) * np.sin(h * z + phi_s)
        return z, delta_new
    return kick


def rf_kick_second_harmonic(V, h, phi_s, beta, E0, a=0.5, phi2=np.pi):
    """
    Kick map with a second harmonic RF system.
    'a' is the voltage ratio V2 / V1.
    """
    def kick(z, delta):
        rf1 = np.sin(h * z + phi_s)
        rf2 = a * np.sin(2 * h * z + phi2)
        delta_new = delta + (V / (beta**2 * E0)) * (rf1 + rf2)
        return z, delta_new
    return kick


def rf_drift_inverse(eta):
    """
    Inverse of the drift map: reverse sign of eta.
    """
    def drift(z, delta):
        z_new = z - eta * delta
        return z_new, delta
    return drift


def rf_kick_inverse(V, h, phi_s, beta, E0):
    """
    Inverse of the RF kick map: reverse sign of the kick.
    """
    def kick(z, delta):
        delta_new = delta - (V / (beta**2 * E0)) * np.sin(h * z + phi_s)
        return z, delta_new
    return kick
