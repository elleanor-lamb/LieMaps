import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# --- Parameters ---
h = 1
eta = 0.01
beta = 1.0
Es = 1e9        # eV
q = 1.6e-19     # C

# RF parameters
V1 = 1e3         # Fundamental RF voltage [eV]
V2 = 0.3 * V1    # Second harmonic RF voltage [eV]
psi = 0          # Phase offset of second harmonic

# Derived constants
a = (2 * np.pi * h * eta) / (beta**2 * Es)
b1 = V1 / (2 * np.pi)   # scaled for Lie map (remove q for eV units)
b2 = V2 / (2 * np.pi)

# --- Symbolic variables ---
phi_sym, dE_sym = sp.symbols('phi dE')

# Hamiltonians
H_drift = (a / 2) * dE_sym**2
H_kick = - b1 * sp.cos(phi_sym) - b2 * sp.cos(2 * phi_sym + psi)

# Poisson bracket / Lie operator
def lie_bracket(f, g):
    df_phi = sp.diff(f, phi_sym)
    df_dE = sp.diff(f, dE_sym)
    dg_phi = sp.diff(g, phi_sym)
    dg_dE = sp.diff(g, dE_sym)
    return df_phi * dg_dE - df_dE * dg_phi

# Lie map: exp(:f:) g ≈ g + {g, f}
def lie_map(f, vars_):
    return [g + lie_bracket(f, g) for g in vars_]

# Symplectic integrator: half-drift, full kick, half-drift
half_drift_map = lie_map(H_drift / 2, [phi_sym, dE_sym])
kick_map = lie_map(H_kick, half_drift_map)
full_map = lie_map(H_drift / 2, kick_map)

# Simplify and lambdify
phi_next_expr, dE_next_expr = [sp.simplify(expr) for expr in full_map]
lie_map_func = sp.lambdify((phi_sym, dE_sym), [phi_next_expr, dE_next_expr], 'numpy')

# --- Tracking ---
def generate_particles(n_particles):
    phi0 = np.random.uniform(-np.pi, np.pi, n_particles)        # rad
    dE0 = np.random.uniform(-2e6, 2e6, n_particles)         # eV
    return phi0, dE0

def track_lie_map(phi0, dE0, n_turns):
    phi, dE = phi0.copy(), dE0.copy()
    trajectories = []

    for _ in range(n_turns):
        phi, dE = lie_map_func(phi, dE)

        trajectories.append((phi.copy(), dE.copy()))

    return trajectories

def plot_trajectories(trajectories):
    plt.figure(figsize=(8, 6))
    for i in range(len(trajectories[0][0])):
        phi_traj = [step[0][i] for step in trajectories]
        dE_traj = [step[1][i] for step in trajectories]
        plt.plot(phi_traj, dE_traj, lw=0.7)
    plt.xlabel("Phase ϕ [rad]")
    plt.ylabel("Energy Deviation ΔE [eV]")
    plt.title("Particle Trajectories in Longitudinal Phase Space\nwith 2nd Harmonic RF")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# --- Potential well plot ---
def plot_potential_well():
    phi = np.linspace(0, 2 * np.pi, 1000)
    U = - V1 * np.cos(phi) - V2 * np.cos(2 * phi + psi)

    plt.figure(figsize=(8, 5))
    plt.plot(phi, U)
    plt.xlabel(r'Phase $\phi$ [rad]')
    plt.ylabel('Potential Energy (eV)')
    plt.title('RF Potential Well with Fundamental + 2nd Harmonic')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# --- Run ---
n_particles = 200
n_turns = 5000
phi0, dE0 = generate_particles(n_particles)
trajectories = track_lie_map(phi0, dE0, n_turns)
plot_potential_well()
plot_trajectories(trajectories)
