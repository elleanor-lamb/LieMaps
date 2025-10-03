import numpy as np
import matplotlib.pyplot as plt

# --- Machine and Beam Parameters ---
q = 1.6               # Charge [C]
V_rf = 1e3                # RF voltage [V]
h = 1                     # Harmonic number
eta = 0.5               # Slip factor
beta = 1.0                # Relativistic beta
Es = 1e9                  # Synchronous energy [eV]

# Precompute constant in drift equation
drift_coef = (2 * np.pi * h * eta) / (beta**2 * Es)

# --- Initial Conditions ---
def generate_particles(n_particles):
    phi0 = np.random.uniform( size=n_particles)*2*np.pi        # [rad]
    dE0 = np.random.normal(size=n_particles)          # [eV]
    return phi0, dE0

# --- Tracking Function ---
def track_particles(phi0, dE0, n_turns):
    n_particles = len(phi0)
    phi, dE = phi0.copy(), dE0.copy()
    trajectories = []

    for _ in range(n_turns):
        # Kick: update energy
        dE = dE + q * V_rf * np.sin(phi)

        # Drift: update phase using updated energy
        phi = phi + drift_coef * dE
        #phi = np.mod(phi + np.pi, 2 * np.pi) - np.pi  # wrap phase to [-π, π)
        #phi = np.mod(phi, 2 * np.pi)  # phase wrapped [0, 2pi)

        trajectories.append((phi.copy(), dE.copy()))

    return trajectories

def plot_trajectories(trajectories):
    plt.figure(figsize=(8, 6))
    for i in range(len(trajectories[0][0])):  # per particle
        phi_traj = [step[0][i] for step in trajectories]
        dE_traj = [step[1][i] for step in trajectories]
        plt.plot(phi_traj, dE_traj, lw=0.5)
    plt.xlabel("Phase ϕ [rad]")
    plt.ylabel("Energy Deviation ΔE [eV]")
    plt.title("Longitudinal Phase Space Trajectories")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

n_particles = 200
n_turns = 3000
phi0, dE0 = generate_particles(n_particles)
trajectories = track_particles(phi0, dE0, n_turns)
plot_trajectories(trajectories)


