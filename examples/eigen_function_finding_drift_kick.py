# Example usage with your RF maps (make sure to define these):
from src.maps import rf_drift_inverse, rf_kick_inverse, rf_drift, rf_kick
from src.eigen_function_finder import iterative_eigenfunction_finder_multi_top2harmonics
import matplotlib.pyplot as plt
import numpy as np

# --- Parameters
eta = 5e-3
V = 1e3
h = 1
phi_s = np.pi
beta = 0.999
E0 = 450e9

# --- Build symplectic map
drift = rf_drift(eta)
kick = rf_kick(V, h, phi_s, beta, E0)
kick_inv = rf_kick_inverse(V, h, phi_s, beta, E0)
drift_inv = rf_drift_inverse(eta)

forward_map = [kick, drift]
inverse_map = [drift_inv, kick_inv]

N_particles = 100
angle = np.linspace(0, 2*np.pi, N_particles, endpoint=False)
radius = 0.1
z0 = radius * np.cos(angle)
dp0 = radius  * np.sin(angle)


trial_z, trial_dp, errors = iterative_eigenfunction_finder_multi_top2harmonics(
    z0, dp0,
    forward_map,
    inverse_map,
    N_turns=1024,
    num_harmonics=20,
    tol=1e-9,
    max_iter=20,
    verbose=True
)

plt.clf()
plt.semilogy(errors, marker='o')
plt.xlabel("Iteration")
plt.ylabel("Backtracking error")
plt.title("Convergence of eigenfunction iteration")
plt.grid(True)
plt.show()
