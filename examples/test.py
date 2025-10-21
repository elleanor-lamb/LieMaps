import numpy as np
import matplotlib.pyplot as plt

def drift_map(z, delta, eta=1.0):
    return z + eta * delta, delta

def kick_map(z, delta, strength=0.1):
    return z, delta + strength * np.sin(z)

# Initial condition near origin
z0 = 0.1
delta0 = 0.0

z, delta = z0, delta0
z_list = []
delta_list = []

for _ in range(500):
    z, delta = drift_map(z, delta)
    z, delta = kick_map(z, delta)
    z_list.append(z)
    delta_list.append(delta)

# Wrap z to [-π, π]
def wrap_phase(z, center=0.0):
    """Wrap phase to be centered around `center` ∈ [-π, π)."""
    return (z - center + np.pi) % (2 * np.pi) - np.pi + center


z_wrapped = wrap_phase(z_list[:, -1], center=0.0)

plt.plot(z_list, delta_list)
plt.xlabel("z [rad]")
plt.ylabel("delta")
plt.axis("equal")
plt.title("Phase space trajectory")
plt.grid(True)
plt.show()
