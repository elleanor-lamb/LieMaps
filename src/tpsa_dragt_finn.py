import tpsa

def init_da(order=5, nvar=2, nvec=100):
    tpsa.da_init(order, nvar, nvec)
    return tpsa.base()  # da[0] = q, da[1] = p


# utils.py
def poisson_bracket(f, g):
    """Compute the Poisson bracket {f, g} = df/dq * dg/dp - df/dp * dg/dq"""
    df_dq = tpsa.da_der(f, 0)
    df_dp = tpsa.da_der(f, 1)
    dg_dq = tpsa.da_der(g, 0)
    dg_dp = tpsa.da_der(g, 1)
    return df_dq * dg_dp - df_dp * dg_dq


def extract_order_terms(da_vec, order):
    """Extract only the terms of a given total order from a DA vector"""
    result = tpsa.base()[0] * 0.0  # zeroed DA vector
    for i in range(da_vec.length()):
        orders, coeff = da_vec.index_element(i)
        if sum(orders) == order:
            result += coeff * monomial_from_orders(orders)
    return result

def da_power(da_var, power):
    if power == 0:
        return 1.0
    result = da_var
    for _ in range(1, power):
        result *= da_var
    return result

def monomial_from_orders(orders):
    da = tpsa.base()
    term = 1.0
    for i, o in enumerate(orders):
        term *= da_power(da[i], o)
    return term

def make_lie_generator(q_map, p_map, order):
    q_h = extract_order_terms(q_map, order)
    p_h = extract_order_terms(p_map, order)

    # Construct approximate inverse via brute force (manual guess)
    # Better: symbolic solver (but TPSA doesn't do this natively)
    # Here, just return a guess:
    return (q_h * 0.0 + p_h * 0.0)  # TODO: replace with manual derivation or symbolic tool

def apply_lie_generator(z_vec, f):
    """Applies e^{:f:} z ≈ z + {z, f}"""
    return [
        z_vec[0] + poisson_bracket(z_vec[0], f),
        z_vec[1] + poisson_bracket(z_vec[1], f)
    ]

def print_da(name, da_vec):
    print(f"--- {name} ---")
    for i in range(da_vec.length()):
        orders, coeff = da_vec.index_element(i)
        if abs(coeff) > 1e-12:
            print(f"{coeff:+.5e} * q^{orders[0]} * p^{orders[1]}")


