# tracking_example.py
from src.tpsa_dragt_finn import init_da, apply_lie_generator, make_lie_generator
import tpsa

da = init_da(order=5)
q = da[0]
p = da[1]

# Define your non-symplectic map
q_map = q + 0.01 * p
p_map = p + 0.02 * q * q

# Construct f2 (manually or using your algorithm)
# For now, let’s say f2 = a * q^2
f2 = 0.01 * da[0] * da[0] / 2

# Apply Lie map
q_sym, p_sym = apply_lie_generator([q, p], f2)

# Print for verification
print('done')
tpsa.print(q_sym)
tpsa.print(p_sym)

