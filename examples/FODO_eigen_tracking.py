from src.eigen_function_kicks_drifts import drift_FODO, quad_FODO
import numpy as np
from matplotlib import pyplot as plt
import nafflib


# FODO cell parameters
L_d = 1.0  # drift length (meters)
f_foc = 2.0  # focusing quad focal length (positive)
f_def = -2.0  # defocusing quad focal length (negative)

# Construct cell matrix
M_foc = quad_FODO(f_foc)
M_def = quad_FODO(f_def)
M_d = drift_FODO(L_d)

M_cell = M_d @ M_def @ M_d @ M_foc

# Inverse map (matrix inverse)
M_cell_inv = np.linalg.inv(M_cell)

# Tracking function
def track_fodo(x0, xp0, M, N_turns):
    coords = np.zeros((N_turns, 2))
    coords[0] = [x0, xp0]
    for i in range(1, N_turns):
        coords[i] = M @ coords[i-1]
    return coords[:, 0], coords[:, 1]

# Backtracking function
def backtrack_fodo(x_arr, xp_arr, M_inv):
    N_turns = len(x_arr)
    coords_back = np.zeros((N_turns, 2))
    coords_back[-1] = [x_arr[-1], xp_arr[-1]]
    for i in reversed(range(N_turns-1)):
        coords_back[i] = M_inv @ coords_back[i+1]
    return coords_back[:, 0], coords_back[:, 1]

# Example initial condition (small amplitude oscillation)
x0, xp0 = 1e-3, 0.0  # 1 mm displacement, zero angle

N_turns = 1000
x_hist, xp_hist = track_fodo(x0, xp0, M_cell, N_turns)

plt.plot(x_hist, xp_hist)
plt.xlabel("x [m]")
plt.ylabel("x' [rad]")
plt.title("FODO Phase Space Trajectory")
plt.grid()
plt.axis('equal')
plt.show()
print('I am here')
# Compute tune from matrix trace
mu = np.arccos(np.trace(M_cell)/2)/ (2 * np.pi)  # tune in [0, 0.5]
print(f"Estimated tune from matrix: {mu:.5f}")

# --- Parameters ---
N_turns = 2000
max_harmonics = 5
window_order = 4


# Initial guess: small amplitude circular motion at tune from matrix
A_guess = [1e-3]
Q_guess = [mu]
t_arr = np.arange(N_turns)

x_guess = A_guess[0] * np.cos(2 * np.pi * Q_guess[0] * t_arr)
xp_guess = A_guess[0] * np.sin(2 * np.pi * Q_guess[0] * t_arr)

all_forward_tracks = []
all_backward_tracks = []
x_initial = []
x_initial.append((x_guess, xp_guess))


# --- Iterative backtracking with increasing harmonics ---
for n_harm in range(1, max_harmonics + 1):
    print(f"\nIteration {n_harm} (extracting {n_harm} harmonics)")

    # 1. Forward track: start from first point of guess
    x0, xp0 = x_guess[0], xp_guess[0]
    x_hist, xp_hist = track_fodo(x0, xp0, M_cell, N_turns)

    all_forward_tracks.append((x_hist, xp_hist))

    # 2. Extract n harmonics
    A, Q = nafflib.harmonics(x_hist, xp_hist, n_harm, window_order=window_order)

    #for i, (A, Q, ph) in enumerate(zip(amps, tunes, phases), 1):
    #    print(f"  Harmonic {i}: Amplitude={A:.4e}, Tune={Q:.6f}, Phase={ph:.4f}")

    # 3. Reconstruct sum of n harmonics
    A_use = A[:n_harm]
    Q_use = Q[:n_harm]
    x_rec, xp_rec = nafflib.generate_signal(A_use, Q_use, t_arr)

    # 4. Backtrack reconstructed signal using inverse map
    x_back, xp_back = backtrack_fodo(x_rec, xp_rec, M_cell_inv)
    all_backward_tracks.append((x_back, xp_back))

    # 5. Compute error at starting point
    err = np.sqrt((x_back[0] - x_guess[0])**2 + (xp_back[0] - xp_guess[0])**2)
    print(f"Backtracking error at start point: {err:.3e}")

    # 6. Update guess (weighted average)
    x_guess = x_guess + (x_back - x_guess)
    xp_guess = xp_back + (xp_back - xp_guess)

    x_initial.append((x_guess.copy(), xp_guess.copy()))
    # Plot phase space for this iteration
    plt.figure(figsize=(7,6))
    plt.plot(x_hist, xp_hist, label="Forward track")
    plt.plot(x_rec, xp_rec, '--', label="Reconstructed sum of harmonics")
    plt.plot(x_back, xp_back, ':', label="Backtracked signal")
    plt.xlabel("x [m]")
    plt.ylabel("x' [rad]")
    plt.title(f"Iteration {n_harm}")
    plt.legend()
    plt.grid()
    plt.axis('equal')
    plt.show()

#print("\nFinal converged phasors:")
#for i, (A, Q, ph) in enumerate(zip(amps, tunes, phases), 1):
#    print(f"Harmonic {i}: Amplitude={A:.4e}, Tune={Q:.6f}, Phase={ph:.4f}")

plt.figure(figsize=(10, 8))
for idx, (x, xp) in enumerate(all_forward_tracks):
    plt.plot(x, xp, label=f'Iteration {idx+1}', alpha=0.7)

plt.xlabel("x")
plt.ylabel("xp")
plt.title("All Forward Tracks per Iteration")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 8))
for idx, (x, xp) in enumerate(all_backward_tracks):
    plt.plot(x, xp, label=f'Iteration {idx+1}', alpha=0.7)

plt.xlabel("x")
plt.ylabel("xp")
plt.title("All Backward Tracks per Iteration")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

