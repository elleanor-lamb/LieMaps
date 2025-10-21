import numpy as np
import matplotlib.pyplot as plt
import nafflib
from src.eigen_function_kicks_drifts import *

# Forward and inverse maps
eta = 0.5
V = 1e7
h = 1
phi_s = 0
beta = 0.999
E0 = 450e9
phi2 = 0.2
a = 0.5

drift_map = rf_drift(eta)
kick_map = rf_kick_second_harmonic(V, h, phi_s, beta, E0, a, phi2)
kick_inv_map = rf_kick_second_harmonic_inv(V, h, phi_s, beta, E0, a, phi2)
drift_inv_map = rf_drift_inv(eta)
forward_map = [drift_map, kick_map]
inverse_map = [kick_inv_map, drift_inv_map]  # inverse order

def forward_track(z0, delta0, forward_map, N_turns):
    z = np.copy(z0)
    delta = np.copy(delta0)
    z_hist = np.zeros(N_turns)
    delta_hist = np.zeros(N_turns)
    for t in range(N_turns):
        for m in forward_map:
            z, delta = m(z, delta)
        z_hist[t] = z
        delta_hist[t] = delta
    return z_hist, delta_hist

def backward_track(z_arr, delta_arr, inverse_map):
    N_turns = len(z_arr)
    z_back = np.copy(z_arr)
    delta_back = np.copy(delta_arr)
    for t in range(N_turns-1, -1, -1):
        for m in inverse_map:
            z_back[t], delta_back[t] = m(z_back[t], delta_back[t])
        z_back[t] = z_back[t] % (2 * np.pi)  # wrap in [0, 2pi)
    return z_back, delta_back

# Parameters
N_turns = 20000
tol = 1e-8
max_harmonics = 8  # Build up to 3 harmonics
window_order = 4

# Initial guess: single circle phasor with amplitude 0.1 and tune guess 0.2
Q_guess = [0.1]
A_guess = [0.3]  # amplitude

Q_initial = Q_guess.copy()
A_initial = A_guess.copy()

# Start from z, delta arrays representing the initial guess phasor signal:
t_arr = np.arange(N_turns)
z_guess = A_guess[0] * np.cos(2 * np.pi * Q_guess[0] * t_arr)
delta_guess = A_guess[0] * np.sin(2 * np.pi * Q_guess[0] * t_arr)

# Wrap initial phase
z_guess = (z_guess + np.pi) % (2 * np.pi)


all_forward_tracks = []
all_inverse_tracks = []

for n_harm in range(1, max_harmonics + 1):
    print(f"\n=== Harmonic {n_harm} ===")

    # 1. Forward track the current guess (take last z, delta as starting points)
    z0 = z_guess[0]
    delta0 = delta_guess[0]
    z_hist, delta_hist = forward_track(z0, delta0, forward_map, N_turns)
    all_forward_tracks.append((z_hist.copy(), delta_hist.copy()))

    # 2. Extract harmonics from forward track
    A, Q = nafflib.harmonics(z_hist, delta_hist, n_harm, window_order=window_order)

    print(f"Harmonics extracted (n={n_harm}):")
    for i in range(n_harm):
        print(f"  Harmonic {i + 1}: Amplitude = {A[i]:.5e}, Tune = {Q[i]:.6f}")

    # PLOT phasors amplitude vs tune
    # plt.figure(figsize=(8, 6))
    # plt.stem(Q[:n_harm], A[:n_harm], basefmt=" ", )
    # plt.xlabel("Tune (Q)")
    # plt.ylabel("Amplitude")
    # plt.title(f"Phasor amplitudes and tunes at iteration {n_harm}")
    # plt.grid(True)
    # plt.show()

    # 3. Reconstruct full signal using all extracted harmonics so far
    # Take first n_harm harmonics from extracted ones
    A_use = A[:n_harm]
    Q_use = Q[:n_harm]
    z_rec, delta_rec = nafflib.generate_signal(A_use, Q_use, t_arr)

    # Wrap phase
    z_rec = z_rec % (2 * np.pi)

    # 4. Backtrack reconstructed signal
    z_back, delta_back = backward_track(z_rec, delta_rec, inverse_map)
    all_inverse_tracks.append((z_back.copy(), delta_back.copy()))

    # 5. Compute error on starting point
    err = np.sqrt((z_back[0] - z_guess[0])**2 + (delta_back[0] - delta_guess[0])**2)
    print(f"Backtracking error: {err:.3e}")

    # 6. Update guess for next iteration:
    # Instead of replacing guess entirely, sum old and diff (could also do weighted update)
    z_guess = z_guess + (z_back - z_guess)
    delta_guess = delta_guess + (delta_back - delta_guess)

    # Wrap phase after update
    z_guess = z_guess % (2 * np.pi)

    # Plot iteration results
    # plt.figure(figsize=(8,6))
    # plt.plot(z_hist, delta_hist, label="Forward track")
    # plt.plot(z_rec, delta_rec, label="Reconstructed signal", alpha=0.5)
    # plt.plot(z_back, delta_back, label="Backtracked signal", linestyle="--", alpha=0.5)
    # plt.xlabel("z [rad]")
    # plt.ylabel("delta")
    # plt.title(f"Iteration with {n_harm} harmonics")
    # plt.legend()
    # plt.grid()
    # plt.show()

plt.figure(figsize=(10, 8))
for idx, (z_hist, delta_hist) in enumerate(all_forward_tracks):
    plt.plot(z_hist, delta_hist, label=f'Harmonic {idx+1}', alpha=0.7)

plt.xlabel("z [rad]")
plt.ylabel("delta")
plt.title("All Forward Tracks per Iteration")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 8))
for idx, (z_hist, delta_hist) in enumerate(all_inverse_tracks):
    plt.plot(z_hist, delta_hist, label=f'Harmonic {idx+1}', alpha=0.7)

plt.xlabel("z [rad]")
plt.ylabel("delta")
plt.title("All Backward Tracks per Iteration")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()


# save final array z, delta to .npy with the initial amplitude and tune given
# Prepare final results
final_data = {
    "z_final": z_back,                  # Final z after last iteration
    "delta_final": delta_back,          # Final delta after last iteration
    "A_initial": A_initial,                 # Initial amplitudes used in final reconstruction
    "Q_initial": Q_initial                  # Initial tunes used in final reconstruction
}

# Save to .npy
np.save(f"./results/{A_initial}_{Q_initial}_final_results.npy", final_data)

