import numpy as np
import matplotlib.pyplot as plt
import os

results_dir = "results/"
result_files = [f for f in os.listdir(results_dir) if f.endswith('_final_results.npy')]
result_files.sort()

plt.figure(figsize=(12, 8))
for filename in result_files:
    filepath = os.path.join(results_dir, filename)

    # Load data
    loaded_data = np.load(filepath, allow_pickle=True).item()
    z_final = loaded_data["z_final"]
    delta_final = loaded_data["delta_final"]
    A_initial = loaded_data["A_initial"]
    Q_initial = loaded_data["Q_initial"]

    A_str = ", ".join([f"{a:.2e}" for a in A_initial[:2]])  # limit to first 2 for readability
    Q_str = ", ".join([f"{q:.4f}" for q in Q_initial[:2]])

    label = f"{filename}\nA: [{A_str}] Q: [{Q_str}]"
    plt.plot(z_final, delta_final, label=label, alpha=0.8)

plt.title("Final z vs delta from all result files")
plt.xlabel("z [rad]")
plt.ylabel("delta")
plt.grid(True)
plt.tight_layout()
plt.show()
