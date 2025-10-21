import numpy as np
import matplotlib.pyplot as plt
import nafflib

def iterative_eigenfunction_finder_multi_top2harmonics(
    z0, dp0,
    forward_map, inverse_map,
    N_turns=20000, num_harmonics=50,
    tol=1e-8, max_iter=20,
    window_order=4,
    verbose=True
):
    N_particles = len(z0)
    trial_z = np.array(z0)
    trial_dp = np.array(dp0)
    errors = []

    # Plot initial topology
    plt.figure(figsize=(7, 7))
    plt.title("Iterative eigenfunction finder - topology evolution")
    plt.xlabel("z")
    plt.ylabel("dp")
    plt.axis('equal')
    plt.grid(True)
    plt.plot(trial_z, trial_dp, 'ko', label='Initial trial')

    for iteration in range(max_iter):
        # --- 1. Forward tracking ---
        z_hist = np.zeros((N_particles, N_turns))
        dp_hist = np.zeros((N_particles, N_turns))
        z = np.copy(trial_z)
        dp = np.copy(trial_dp)

        for t in range(N_turns):
            for m in forward_map:
                z_new = np.empty_like(z)
                dp_new = np.empty_like(dp)
                for i in range(N_particles):
                    z_new[i], dp_new[i] = m(z[i], dp[i])
                z, dp = z_new, dp_new
            z_hist[:, t] = z
            dp_hist[:, t] = dp

        # --- 2. Extract and select top-2 harmonics ---
        A_list = []
        Q_list = []
        for i in range(N_particles):
            s = z_hist[i] - 1j * dp_hist[i]
            A_all, Q_all = nafflib.harmonics(s.real, s.imag, num_harmonics, window_order=window_order)

            # Get top-2 by amplitude
            top4_idx = np.argsort(np.abs(A_all))[:]
            top4_idx = top4_idx[np.argsort(Q_all[top4_idx])]  # sort by frequency

            A_top2 = A_all[top4_idx]
            Q_top2 = Q_all[top4_idx]

            A_list.append(A_top2)
            Q_list.append(Q_top2)

        # --- 3. Reconstruct signal using only top 2 harmonics ---
        z_rec = np.zeros_like(z_hist)
        dp_rec = np.zeros_like(dp_hist)
        t_arr = np.arange(N_turns)
        for i in range(N_particles):
            z_r, dp_r = nafflib.generate_signal(A_list[i], Q_list[i], t_arr)
            z_rec[i] = z_r
            dp_rec[i] = dp_r

        # --- 4. Backtrack reconstructed signal ---
        z_back = np.copy(z_rec)
        dp_back = np.copy(dp_rec)
        for t in range(N_turns - 1, -1, -1):
            for m in inverse_map:
                for i in range(N_particles):
                    z_back[i, t], dp_back[i, t] = m(z_back[i, t], dp_back[i, t])

        # --- 5. Compute error at start ---
        err_vec = np.sqrt((z_back[:, 0] - trial_z)**2 + (dp_back[:, 0] - trial_dp)**2)
        max_err = np.max(err_vec)
        mean_err = np.mean(err_vec)
        errors.append(max_err)

        if verbose:
            print(f"Iteration {iteration+1}: max error = {max_err:.3e}, mean error = {mean_err:.3e}")

        # --- 6. Plot topology evolution ---
        if iteration > 18:
            plt.plot(trial_z, trial_dp, 'ro', label='Prev trial' if iteration == 1 else "")
            plt.plot(z_back[:, 0], dp_back[:, 0], 'bx', label='Backtracked start' if iteration == 1 else "")
            plt.legend(loc='upper right')
            plt.pause(0.5)
            plt.show()

        # --- 7. Optional: Plot harmonic amplitudes per iteration ---
        # plt.figure(figsize=(10, 4))
        # for h in range(2):
        #     amps = [np.abs(A_list[i][h]) for i in range(N_particles)]
        #     plt.plot(range(N_particles), amps, label=f'Harmonic {h+1}')
        # plt.title(f'Iteration {iteration+1}: Top 2 harmonic amplitudes')
        # plt.xlabel('Particle index')
        # plt.ylabel('Amplitude')
        # plt.grid()
        # plt.legend()
        # plt.show()

        # --- 8. Check convergence ---
        if max_err < tol:
            if verbose:
                print(f"Converged after {iteration+1} iterations.")
            break

        # --- 9. Update trial points ---
        correction_z = z_back[:, 0] - trial_z
        correction_dp = dp_back[:, 0] - trial_dp

        trial_z = trial_z + correction_z
        trial_dp = trial_dp + correction_dp

    return trial_z, trial_dp, errors
