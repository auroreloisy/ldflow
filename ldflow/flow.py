import numpy as np
from scipy.interpolate import RegularGridInterpolator


def tgv_2d():
    """Taylor-Green vortices (2D stationary, invariant along z)"""
    def vector_field(t, u):
        x, y, z, w = u.T    # w is just a hack (requires even number of vars)
        v_x = np.cos(x) * np.sin(y)
        v_y = - np.sin(x) * np.cos(y)
        v_z = np.zeros_like(v_x)
        v_w = np.ones_like(v_x)  # hack
        v = np.column_stack([v_x, v_y, v_z, v_w])
        return v
    return vector_field


def abc(a=np.sqrt(3), b=np.sqrt(2), c=1):
    """ABC flow (3D stationary)"""
    def vector_field(t, u):
        """
        Parameters
        ----------
        t : float
            fixed time-point of vector field, for all points in phase space.

        u : array_like, shape(n,)
            Points in phase space.

        Returns
        -------
        v : array_like, shape(n,)
            Vector field corresponding to points u, in phase space at time t.
        """
        x, y, z, w = u.T    # w is just a hack (requires even number of vars)
        v_x = a * np.sin(z) + c * np.cos(y)
        v_y = b * np.sin(x) + a * np.cos(z)
        v_z = c * np.sin(y) + b * np.cos(x)
        v_w = np.ones_like(w)  # hack
        v = np.column_stack([v_x, v_y, v_z, v_w])
        return v

    return vector_field


def turb_frozen(n_grid=128, file='v006018.csv', print_info=False):
    """Frozen 3D HIT"""
    # csv are columns: vx, vy, vz with a header
    dat = np.genfromtxt('./data/turb_frozen/' + file, delimiter=',', skip_header=1)
    assert dat.shape[0] == n_grid ** 3
    field = np.zeros([n_grid + 1, n_grid + 1, n_grid + 1, 3])
    field[:n_grid, :n_grid, :n_grid, :] = dat.reshape([n_grid, n_grid, n_grid, 3])
    del dat
    # add periodic BC (copy slices)
    field[:n_grid, :n_grid, n_grid, :] = field[:n_grid, :n_grid, 0, :]
    field[:n_grid, n_grid, :, :] = field[:n_grid, 0, :, :]
    field[n_grid, :, :, :] = field[0, :, :, :]
    # domain is [0, 2*pi]^3
    x = np.linspace(0, 2*np.pi, n_grid + 1)
    y = np.linspace(0, 2*np.pi, n_grid + 1)
    z = np.linspace(0, 2*np.pi, n_grid + 1)
    points = (x, y, z)
    # interpolating function
    method = 'linear'
    interp_vx = RegularGridInterpolator(points, field[:, :, :, 0], method=method)
    interp_vy = RegularGridInterpolator(points, field[:, :, :, 1], method=method)
    interp_vz = RegularGridInterpolator(points, field[:, :, :, 2], method=method)

    def vector_field(t, u):
        if print_info:
            print('t =', t)

        x, y, z, w = u.T
        xg = x % (2*np.pi)
        yg = y % (2*np.pi)
        zg = z % (2*np.pi)

        v_x = interp_vx((xg, yg, zg))
        v_y = interp_vy((xg, yg, zg))
        v_z = interp_vz((xg, yg, zg))
        v_w = np.ones_like(v_z)  # hack
        v = np.column_stack([v_x, v_y, v_z, v_w])

        return v

    return vector_field
