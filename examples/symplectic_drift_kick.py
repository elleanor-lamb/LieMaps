import numpy as np
import matplotlib.pyplot as plt
from src.symplectic_maps_rf import rf_kick, rf_drift

def wrap_phase(z):
    return z % (2 * np.pi)

def simulate_and_plot(maps, z0, delta0, n_turns=100, plot_every=20):
    z = np.copy(z0)
    delta = np.copy(delta0)

    plt.figure(figsize=(8, 6))

    for turn in range(n_turns):
        for m in maps:
            z, delta = m.apply(z, delta)

        z = wrap_phase(z)
        if turn % plot_every == 0 or turn == n_turns - 1:
            plt.scatter(z, delta, s=1, alpha=0.5, label=f"Turn {turn}")

    plt.xlabel("z [rad]")
    plt.ylabel("delta [arb. units]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    eta = 5
    V = 1e7
    h = 1
    phi_s = 0 # synchronous phase
    beta = 0.999
    E0 = 450e9

    drift_map = rf_drift(eta)
    kick_map = rf_kick(V, h, phi_s, beta, E0)
    one_turn_map = [drift_map, kick_map]

    n_particles = 10
    z0 = np.random.uniform(0, 2*np.pi, n_particles) #  [0, 2pi]
    delta0 = np.random.uniform(-0.001, 0.001, n_particles)

    simulate_and_plot(one_turn_map, z0, delta0, n_turns=1000, plot_every=1)
