def run_naff_tracking(A_guess, Q_guess, run_id=None):
    import numpy as np
    import nafflib
    import os
    from src.eigen_function_kicks_drifts import rf_drift, rf_drift_inv, rf_kick_second_harmonic, rf_kick_second_harmonic_inv

    os.makedirs("results", exist_ok=True)

    # Parameters
    N_turns = 20000
    tol = 1e-8
    max_harmonics = 10
    window_order = 4

    # Maps
    eta = 0.5
    V = 1e7
    h1 = 1
    h2 = 2
    phi_s = 0
    beta = 0.999
    E0 = 450e9
    phi2 = 0.3
    a = 0.7

    drift_map = rf_drift(eta)
    kick_map = rf_kick_second_harmonic(V, h1, h2, phi_s, beta, E0, a, phi2)
    kick_inv_map = rf_kick_second_harmonic_inv(V, h1, h2, phi_s, beta, E0, a, phi2)
    drift_inv_map = rf_drift_inv(eta)
    forward_map = [drift_map, kick_map]
    inverse_map = [kick_inv_map, drift_inv_map]

    def forward_track(z0, delta0, forward_map, N_turns):
        z, delta = np.copy(z0), np.copy(delta0)
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
        z_back, delta_back = np.copy(z_arr), np.copy(delta_arr)
        for t in range(N_turns-1, -1, -1):
            for m in inverse_map:
                z_back[t], delta_back[t] = m(z_back[t], delta_back[t])
            z_back[t] = z_back[t] % (2 * np.pi)
        return z_back, delta_back

    # Start with single phasor signal guess
    t_arr = np.arange(N_turns)
    z_guess = A_guess * np.cos(2 * np.pi * Q_guess * t_arr)
    delta_guess = A_guess * np.sin(2 * np.pi * Q_guess * t_arr)
    z_guess = (z_guess + np.pi) % (2 * np.pi)

    all_forward_tracks = []
    all_inverse_tracks = []

    for n_harm in range(1, max_harmonics + 1):
        print(f"\n=== Harmonic {n_harm} | A={A_guess:.3f}, Q={Q_guess:.3f} ===")

        z0 = z_guess[0]
        delta0 = delta_guess[0]
        z_hist, delta_hist = forward_track(z0, delta0, forward_map, N_turns)
        all_forward_tracks.append((z_hist.copy(), delta_hist.copy()))

        A, Q = nafflib.harmonics(z_hist, delta_hist, n_harm, window_order=window_order)

        A_use = A[:n_harm]
        Q_use = Q[:n_harm]
        z_rec, delta_rec = nafflib.generate_signal(A_use, Q_use, t_arr)
        z_rec = z_rec % (2 * np.pi)

        z_back, delta_back = backward_track(z_rec, delta_rec, inverse_map)
        all_inverse_tracks.append((z_back.copy(), delta_back.copy()))

        err = np.sqrt((z_back[0] - z_guess[0])**2 + (delta_back[0] - delta_guess[0])**2)
        print(f"Backtracking error: {err:.3e}")

        z_guess = (z_guess + (z_back - z_guess)) % (2 * np.pi)
        delta_guess = delta_guess + (delta_back - delta_guess)

    # Save results
    final_data = {
        "z_final": z_back,
        "delta_final": delta_back,
        "A_initial": [A_guess],
        "Q_initial": [Q_guess]
    }

    if run_id is None:
        run_id = f"{A_guess:.3f}_{Q_guess:.3f}"
    filename = f"results/{run_id}_final_results.npy"
    np.save(filename, final_data)
    print(f"✅ Saved: {filename}")
