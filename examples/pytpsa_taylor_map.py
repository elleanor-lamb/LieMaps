import tpsa
import matplotlib.pyplot as plt

# Initialize TPSA
order = 20
nvar = 2
nvec = 100
tpsa.da_init(order, nvar, nvec)
da = tpsa.base()  # da[0] = q, da[1] = p

q_new = da[0] + da[1]  # Drift: q' = q + p
p_new = da[1] - da[0]  # Kick:  p' = p - q

# Multivariate evaluator
def da_eval(da_vec, values, max_power=20):
    total = 0.0
    n_terms = da_vec.length()
    for i in range(n_terms):
        orders, coeff = da_vec.index_element(i)
        term = coeff
        for var_index, power in enumerate(orders):
            if power > 0:
                if power > max_power:
                    term = 0.0
                    break
                try:
                    term *= values[var_index] ** power
                except OverflowError:
                    term = 0.0
                    break
        total += term
    return total

# Track a particle in (q, p) using the nonlinear symplectic map
N_turns = 100
q_vals = []
p_vals = []
q, p = 0.0001, 0.0001

for _ in range(N_turns):
    q_next = da_eval(q_new, [q, p])
    p_next = da_eval(p_new, [q, p])
    q_vals.append(q_next)
    p_vals.append(p_next)
    q, p = q_next, p_next

# Plot trajectory in phase space
plt.plot(q_vals, p_vals, '.', markersize=2)
plt.xlabel('q')
plt.ylabel('p')
plt.title('Tracking in phase space with TPSA map (q and p dependence)')
plt.grid(True)
plt.axis('equal')
plt.show()

import numpy as np


def symplecticity_test(q_val, p_val, epsilon=1e-6):
    # Evaluate the map at base point
    f0 = [da_eval(q_new, [q_val, p_val]), da_eval(p_new, [q_val, p_val])]

    # Evaluate map at small displacements in q and p
    f_q = [da_eval(q_new, [q_val + epsilon, p_val]), da_eval(p_new, [q_val + epsilon, p_val])]
    f_p = [da_eval(q_new, [q_val, p_val + epsilon]), da_eval(p_new, [q_val, p_val + epsilon])]

    # Compute numerical derivatives (Jacobian columns)
    dq_dq = (f_q[0] - f0[0]) / epsilon
    dq_dp = (f_p[0] - f0[0]) / epsilon
    dp_dq = (f_q[1] - f0[1]) / epsilon
    dp_dp = (f_p[1] - f0[1]) / epsilon

    # Symplecticity condition
    det = dq_dq * dp_dp - dq_dp * dp_dq
    print(f"Determinant (should be 1): {det:.12f}")
    return det


symplecticity_test(0.1, 0.1)
H_vals = [0.5*q_vals[i]**2 + 0.5*p_vals[i]**2 for i in range(N_turns)]
plt.plot(H_vals)
plt.xlabel('Turn')
plt.ylabel('H')
plt.title('Hamiltonian vs Turn (should be flat)')
plt.grid(True)
plt.show()

