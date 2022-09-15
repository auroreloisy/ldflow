import numpy as np


def abc(t, u, parameters=(np.sqrt(3), np.sqrt(2), 1)):
    """
    ABC flow

    Parameters
    ----------
    t : float
        fixed time-point of vector field, for all points in phase space.

    u : array_like, shape(n,)
        Points in phase space.

    parameters : (A, B, C)
        Flow coefficients

    Returns
    -------
    v : array_like, shape(n,)
        Vector field corresponding to points u, in phase space at time t.
    """
    x, y, z, w = u.T    # w is just a hack (requires even number of vars)

    A = parameters[0]
    B = parameters[1]
    C = parameters[2]
    v_x = A * np.sin(z) + C * np.cos(y)
    v_y = B * np.sin(x) + A * np.cos(z)
    v_z = C * np.sin(y) + B * np.cos(x)
    v_w = np.ones_like(w)  # hack
    v = np.column_stack([v_x, v_y, v_z, v_w])
    return v
