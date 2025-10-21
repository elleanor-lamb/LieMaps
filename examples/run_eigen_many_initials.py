
from src.run_eigen_second_order import run_naff_tracking
import numpy as np

# Define lists of amplitudes and tunes to scan
amplitude_list = np.linspace(0, 2, 20)
tune_list = [0.1]

# Loop over all combinations
for A in amplitude_list:
    for Q in tune_list:
        run_id = f"A{A:.2f}_Q{Q:.2f}".replace('.', 'p')  # Safe for filenames
        run_naff_tracking(A, Q, run_id=run_id)