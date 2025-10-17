
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# --- Symbolic Setup ---

# Define symbols
z_sym, delta_sym = sp.symbols('z delta')
eta_val = 0.01       # slip factor
V_rf_val = 0.01      # RF voltage
h_val = 1            # harmonic number

# Hamiltonians
H_drift = delta_sym**2 / (2 * eta_val)
H_kick = V_rf_val * sp.cos(h_val * z_sym)

# Lie bracket (Poisson bracket)
def lie_operator(f, g):
    dg_dz = sp.diff(g, z_sym)
    dg_ddelta = sp.diff(g, delta_sym)
    df_dz = sp.diff(f, z_sym)
    df_ddelta = sp.diff(f, delta_sym)
    return df_dz * dg_ddelta - df_ddelta * dg_dz

# First-order Lie map: exp(:f:) g ≈ g + :f: g
def lie_map(f, vars_):
    mapped = []
    for g in vars_:
        bracket = lie_operator(f, g)
        mapped.append(g + bracket)
    return mapped

# Create symbolic maps
vars_ = [z_sym, delta_sym]
drift_map = lie_map(H_drift, vars_)
kick_map = lie_map(H_kick, drift_map)

# Lambdify for fast numerical evaluation
z_expr, delta_expr = kick_map
map_func = sp.lambdify((z_sym, delta_sym), [z_expr, delta_expr], modules='numpy')

# --- Particle Initialization ---

def generate_particles(n_particles):
    """Generate particles in phase space (z, delta)"""
    # Gaussian cloud
    z0 = np.random.normal(loc=0, scale=0.5, size=n_particles)
    delta0 = np.random.normal(loc=0, scale=0.01, size=n_particles)
    return z0, delta0

# --- Apply Map Over Many Turns ---

def track_particles(z0, delta0, n_turns=10):
    trajectories = []
    z, d = z0.copy(), delta0.copy()
    for _ in range(n_turns):
        z, d = map_func(z, d)
        z = np.mod(z + np.pi, 2 * np.pi) - np.pi
        trajectories.append((z.copy(), d.copy()))
    return trajectories

# --- Plotting ---

def plot_trajectories(trajectories):
    plt.figure(figsize=(8, 6))
    for i in range(len(trajectories[0][0])):  # For each particle
        z_traj = [step[0][i] for step in trajectories]
        d_traj = [step[1][i] for step in trajectories]
        plt.plot(z_traj, d_traj, lw=0.5)
    plt.xlabel('z (longitudinal phase)')
    plt.ylabel('δ (energy deviation)')
    plt.title('Particle Trajectories in Longitudinal Phase Space')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# --- Run Simulation ---

n_particles = 5
n_turns = 200
z0, delta0 = generate_particles(n_particles)
trajectories = track_particles(z0, delta0, n_turns=n_turns)
plot_trajectories(trajectories)

