"""Generate a grid of
(q,p) (q,p) points in phase space,

Evaluate
H(q,p) H(q,p) over the grid,

Compute
fq(H)
"""
import numpy as np
import matplotlib.pyplot as plt

# Parameters
V1 = 1.0          # Fundamental RF amplitude
V2 = 0.4          # Second harmonic amplitude
phi = np.pi / 3   # Phase offset for 2nd harmonic
L = 1.0           # Drift length

# q-Gaussian parameters
q = 1.3
beta = 1.0

# Phase space grid
q_range = (-np.pi, np.pi)
p_range = (-3.0, 3.0)
q_points = 400
p_points = 400

q_vals = np.linspace(*q_range, q_points)
p_vals = np.linspace(*p_range, p_points)
Q, P = np.meshgrid(q_vals, p_vals)

# ---------- Asymmetric Potential Function ----------
def potential(q):
    return -V1 * np.cos(q) - V2 * np.cos(2 * q + phi)

def dV_dq(q):
    return V1 * np.sin(q) + 2 * V2 * np.sin(2 * q + phi)

# ---------- Hamiltonian ----------
def hamiltonian(q, p):
    return 0.5 * p**2 + potential(q)

H = hamiltonian(Q, P)

# ---------- q-Gaussian Distribution ----------
def f_q(H, q, beta):
    arg = 1 - (1 - q) * beta * H
    with np.errstate(invalid='ignore'):
        return np.where(arg > 0, arg**(1 / (1 - q)), 0.0)

F_q = f_q(H, q=q, beta=beta)

# ---------- Symplectic Map (Kick + Drift) ----------
def symplectic_map(q, p):
    # Kick from potential
    p_new = p - dV_dq(q)
    # Drift
    q_new = q + L * p_new
    return q_new, p_new

# ---------- Plot 1: Asymmetric Potential ----------
plt.figure(figsize=(7, 4))
q_line = np.linspace(-np.pi, np.pi, 1000)
V_plot = potential(q_line)
plt.plot(q_line, V_plot)
plt.title("Asymmetric RF Potential $V(q)$")
plt.xlabel("$q$")
plt.ylabel("$V(q)$")
plt.grid(True)
plt.tight_layout()
plt.show()

# ---------- Plot 2: Hamiltonian Contours ----------
plt.figure(figsize=(8,6))
levels_H = np.linspace(np.min(H), np.max(H), 40)
contours_H = plt.contour(Q, P, H, levels=levels_H, cmap='viridis')
plt.clabel(contours_H, inline=True, fontsize=8)
plt.title('Hamiltonian Contours with Asymmetric Potential')
plt.xlabel('$q$')
plt.ylabel('$p$')
plt.grid(True)
plt.tight_layout()
plt.show()

# ---------- Plot 3: q-Gaussian Distribution ----------
plt.figure(figsize=(8,6))
levels_F = np.linspace(np.min(F_q), np.max(F_q), 100)
plt.contourf(Q, P, F_q, levels=levels_F, cmap='plasma')
plt.colorbar(label='$f_q(H)$')
plt.xlabel('$q$')
plt.ylabel('$p$')
plt.title(f'q-Gaussian Distribution $f_q(H)$, q={q}, β={beta}')
plt.grid(True)
plt.tight_layout()
plt.show()

# ---------- Plot 4: Overlay q-Gaussian + Hamiltonian Contours ----------
# ---------- Plot 4: Overlay q-Gaussian + Hamiltonian Contours ----------
plt.figure(figsize=(8,6))
plt.contourf(Q, P, F_q, levels=100, cmap='plasma', alpha=0.9)
contours_H = plt.contour(Q, P, H, levels=20, colors='white', linewidths=0.8)
plt.clabel(contours_H, inline=True, fontsize=8)
plt.colorbar(label='$f_q(H)$')
plt.xlabel('$q$')
plt.ylabel('$p$')
plt.title('q-Gaussian $f_q(H)$ with Hamiltonian Contours')
plt.grid(True)
plt.tight_layout()
plt.show()
