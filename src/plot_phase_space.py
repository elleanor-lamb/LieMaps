import numpy as np
import matplotlib.pyplot as plt

def plot_phase_space_trajectories(
    forward_map,
    z_range=(-5e-3, 5e-3),
    dp_range=(-5e-3, 5e-3),
    grid_size=40,
    N_turns=2000,
    plot_every=1
):
    # 1. Create uniform grid of initial conditions
    z_vals = np.linspace(*z_range, grid_size)
    dp_vals = np.linspace(*dp_range, grid_size)
    zz, ddpp = np.meshgrid(z_vals, dp_vals)
    z0_flat = zz.flatten()
    dp0_flat = ddpp.flatten()
    N_particles = len(z0_flat)

    # 2. Allocate trajectory arrays
    z_hist = np.zeros((N_particles, N_turns))
    dp_hist = np.zeros((N_particles, N_turns))

    # 3. Forward track
    z = np.copy(z0_flat)
    dp = np.copy(dp0_flat)
    for t in range(N_turns):
        for m in forward_map:
            z_new = np.empty_like(z)
            dp_new = np.empty_like(dp)
            for i in range(N_particles):
                z_new[i], dp_new[i] = m(z[i], dp[i])
            z, dp = z_new, dp_new
        z_hist[:, t] = z
        dp_hist[:, t] = dp

    z_wrapped = np.mod(z_hist[:, -1] + np.pi, 2 * np.pi) - np.pi

    plt.figure(figsize=(8, 8))
    plt.scatter(z_wrapped, dp_hist[:, -1], s=10, alpha=0.7)

    plt.xlabel("z [rad]")
    plt.ylabel("dp")
    plt.title(f"Phase space at turn {N_turns} (z wrapped to [-π, π))")
    plt.grid(True)
    plt.axis("equal")
    plt.tight_layout()
    plt.show()

    return z_hist, dp_hist
