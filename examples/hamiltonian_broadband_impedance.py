import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# --- Parameters ---
n_test_particles = 40
n_turns = 900
phi_min, phi_max = -np.pi, np.pi

# Machine parameters
h = 1
eta = 0.01
beta = 1.0
Es = 1e9  # eV
q = 1.0   # normalized charge for RF kick

# Drift coefficient
drift_coef = (2 * np.pi * h * eta) / (beta**2 * Es)

# RF parameters
V_rf = 1e6  # RF voltage amplitude (eV)

# Broadband impedance wake parameters
W0 = 1e7    # amplitude eV
k = 5       # wave number of oscillation
alpha = 3   # damping factor

# --- Background Gaussian distribution parameters ---
phi_center = 0
sigma_phi = 0.3

# --- Test particle initialization ---
# Sample test particles around the distribution center
phi_test = np.random.uniform(-2*np.pi, 0, size=n_test_particles)
dE_test = np.zeros(n_test_particles)

# --- Define wake function ---
def broadband_wake(phi_diff):
    phi_diff = np.mod(phi_diff, 2 * np.pi)
    return W0 * np.sin(k * phi_diff) * np.exp(-alpha * phi_diff)

# --- Define background distribution on grid ---
def background_distribution(phi_vals):
    # Wrapped Gaussian on [0, 2pi)
    # Sum over a few wraps to approximate periodicity
    p = np.zeros_like(phi_vals)
    for m in range(-3, 4):
        p += norm.pdf(phi_vals + 2*np.pi*m, loc=phi_center, scale=sigma_phi)
    return p / np.trapz(p, phi_vals)  # normalize

# --- Compute energy kick from background distribution ---
def compute_energy_kick(phi_particles, bins=500):
    phi_vals = np.linspace(0, 2 * np.pi, bins)
    dist = background_distribution(phi_vals)
    dphi = phi_vals[1] - phi_vals[0]

    # Convolve background distribution with wake
    wake_grid = np.zeros_like(phi_vals)
    for i, ph in enumerate(phi_vals):
        wake_grid[i] = np.sum(dist * broadband_wake(ph - phi_vals)) * dphi

    # Interpolate wake to test particle phases
    kick = np.interp(phi_particles, phi_vals, wake_grid, period=2*np.pi)
    return kick, phi_vals, wake_grid, dist

# --- Tracking ---
trajectories = []
for turn in range(n_turns):
    # Kick: RF voltage kick
    dE_test += q * V_rf * np.sin(phi_test)

    # Kick: broadband impedance wake from background distribution
    dE_test -= compute_energy_kick(phi_test)[0]

    # Drift update: Lie map drift for phase
    phi_test += drift_coef * dE_test

    trajectories.append((phi_test.copy(), dE_test.copy()))

# --- Plot test particle trajectories ---
plt.figure(figsize=(8, 6))
for i in range(n_test_particles):
    phi_traj = [step[0][i] for step in trajectories]
    dE_traj = [step[1][i] for step in trajectories]
    plt.plot(phi_traj, dE_traj, lw=0.8)
plt.xlabel('Phase ϕ [rad]')
plt.ylabel('Energy Deviation ΔE [eV]')
plt.title('Test Particle Trajectories with RF + Broadband Wake')
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Plot effective potential well ---
phi_vals = np.linspace(0, 2 * np.pi, 1000)
dist_vals = background_distribution(phi_vals)
wake_vals = np.array([np.sum(dist_vals * broadband_wake(ph - phi_vals) * (phi_vals[1] - phi_vals[0])) for ph in phi_vals])
U_wake = -np.cumsum(wake_vals) * (phi_vals[1] - phi_vals[0])
U_wake -= U_wake.min()

# Add RF potential: U_rf(phi) = - q V_rf cos(phi)
U_rf = -q * V_rf * np.cos(phi_vals)
U_total = U_rf + U_wake
U_total -= U_total.min()

plt.figure(figsize=(8, 5))
plt.plot(phi_vals, U_total, label='Total Potential Well (RF + Wake)')
plt.plot(phi_vals, U_rf, linestyle='--', label='RF Potential')
plt.plot(phi_vals, U_wake, linestyle=':', label='Wake Potential')
plt.xlabel('Phase ϕ [rad]')
plt.ylabel('Potential Energy (arb. units)')
plt.title('Effective Potential Well with RF + Broadband Wake')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
