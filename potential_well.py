import numpy as np
import matplotlib.pyplot as plt

# Parameters
V_rf = 1.0  # normalized voltage amplitude (arbitrary units)

# Phase array
phi = np.linspace(0, 2 * np.pi, 500)

# Potential well
U = V_rf * np.cos(phi)

# Plot
plt.figure(figsize=(8, 5))
plt.plot(phi, U, label=r'$U(\phi) = - V_{rf} \cos(\phi)$')
plt.xlabel(r'Phase $\phi$ [rad]')
plt.ylabel('Potential Energy (arb. units)')
plt.title('RF Potential Well')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
