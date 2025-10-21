import numpy as np


def rf_drift(eta):
    def drift(z, delta):
        z_new = z + eta * delta
        z_new = z_new % (2 * np.pi)
        return z_new, delta
    return drift

def rf_kick(V, h, phi_s, beta, E0):
    def kick(z, delta):
        delta_new = delta + (V / (beta**2 * E0)) * np.sin(h * z + phi_s)
        return z, delta_new
    return kick

def rf_kick_second_harmonic(V, h1, h2, phi_s, beta, E0, a=0.5, phi2=np.pi):
    def kick(z, delta):
        rf1 = np.sin(h1 * z + phi_s)
        rf2 = a * np.sin(h2 * z + phi2)
        delta_new = delta + (V / (beta**2 * E0)) * (rf1 + rf2)
        return z, delta_new
    return kick

def rf_drift_inv(eta):
    def drift_inv(z, delta):
        z_new = z - eta * delta
        z_new = z_new % (2 * np.pi)
        return z_new, delta
    return drift_inv

def rf_kick_inv(V, h, phi_s, beta, E0):
    def kick_inv(z, delta):
        delta_new = delta - (V / (beta**2 * E0)) * np.sin(h * z + phi_s)
        return z, delta_new
    return kick_inv

def rf_kick_second_harmonic_inv(V, h1,h2, phi_s, beta, E0, a=0.5, phi2=np.pi):
    def kick_inv(z, delta):
        rf1 = np.sin(h1 * z + phi_s)
        rf2 = a * np.sin(h2 * z + phi2)
        delta_new = delta - (V / (beta**2 * E0)) * (rf1 + rf2)
        return z, delta_new
    return kick_inv


#drift matrix
def drift_FODO(L):
    return np.array([[1, L],
                     [0, 1]])

#lens matrix (focusing or defocusing)
def quad_FODO(f):
    return np.array([[1, 0],
                     [-1/f, 1]])



