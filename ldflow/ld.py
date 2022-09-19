import numpy as np
import numpy.ma as ma
from scipy.integrate import solve_ivp
import ldds.base


def compute_lagrangian_descriptor(parameters, vector_field, tau, p_value=0.5, box_boundaries=False, rtol=1.0e-4):
    """
    Returns the values of the LD function from integrated trajectories from initial conditions in phase space.

    Parameters
    ----------
    parameters : dictionary containing
        'section': 'x', 'y' or 'z'
        'value': float
        'n_points': int, number of points in each dimension

    vector_field: function
        flow

    tau : float
        Upper limit of integration.

    p_value : float, optional
        Exponent in Lagrangian descriptor definition.
        0 is the acton-based LD,
        0 < p_value < 1 is the Lp quasinorm,
        1 <= p_value < 2 is the Lp norm LD,
        2 is the arclength LD.
        The default is 0.5.

    box_boundaries : list of 2-tuples, optional
        Box boundaries for escape condition of variable time integration.
        Boundaries are infinite by default.

    rtol : float,
        Relative tolerance of integration step.

    Returns
    -------
    LD : ndarray, shape (Nx, Ny)
        Array of computed Lagrangian descriptor values for all initial conditions.
    """
    # hack for grid_parameters
    if parameters["section"] == "x":
        dims_fixed = [1, 0, 0, 0]
        dims_slice = [0, 1, 1, 0]  # Visualisation slice
    elif parameters["section"] == "y":
        dims_fixed = [0, 1, 0, 0]
        dims_slice = [1, 0, 1, 0]  # Visualisation slice
    elif parameters["section"] == "z":
        dims_fixed = [0, 0, 1, 0]  # Variable ordering (x y p_x p_y)
        dims_slice = [1, 1, 0, 0]  # Visualisation slice
    else:
        raise Exception("section must be x, y or z")
    dims_fixed_values = [parameters["value"]]  # This can also be an array of values

    ax1_min, ax1_max = [0, 2 * np.pi]
    ax2_min, ax2_max = [0, 2 * np.pi]
    N = parameters["n_points"]
    slice_parameters = [[ax1_min, ax1_max, N], [ax2_min, ax2_max, N]]

    grid_parameters = {
        'slice_parameters': slice_parameters,
        'dims_slice': dims_slice,
        'dims_fixed': dims_fixed,
        'dims_fixed_values': dims_fixed_values,
        'momentum_sign': 1,
        'potential_energy': lambda x: 0,
        'energy_level': 0
    }

    # get visualisation slice parameters and Number of DoF
    if type(grid_parameters) == dict:
        # n-DoF systems
        slice_parameters = np.array(grid_parameters['slice_parameters'])  # 2n-D grid
        N_dim = len(grid_parameters['dims_slice'])
    else:
        # 1-DoF systems
        slice_parameters = np.array(grid_parameters)  # 2-D grid
        N_dim = len(slice_parameters)

    # set boundaries for escape-box condition, if not defined
    if not box_boundaries:
        box_boundaries = int(N_dim / 2) * [[-np.infty, np.infty]]  # restricted to configuration space

    # solve initial value problem
    f = lambda t, y: ldds.base.vector_field_flat(t, y, vector_field, p_value, box_boundaries)
    y0, mask = ldds.base.generate_points(grid_parameters)

    # mask y0 values
    if type(mask) == np.ndarray:
        mask_y0 = np.transpose([mask for i in range(N_dim + 1)]).flatten()
        y0 = ma.masked_array(y0, mask=mask_y0)

    solution = solve_ivp(f, [0, tau], y0, t_eval=[tau], rtol=rtol, atol=1.0e-6, first_step=1e-6)

    N_points_slice_axes = slice_parameters[:, -1].astype('int')

    # displacement along the 3 axes
    displacement = []
    for i in range(3):
        d = solution.y[i::N_dim + 1] - y0[i::N_dim + 1, np.newaxis]  # displacement along i-axis
        d = d.reshape(*N_points_slice_axes)  # reshape to 2-D array
        displacement.append(d)

    # lagrangian descriptor
    LD_values = solution.y[N_dim::N_dim + 1]  # values corresponding to LD
    # LD_values[mask] = np.nan #mask LD values for slice
    LD = np.abs(LD_values).reshape(*N_points_slice_axes)  # reshape to 2-D array

    if p_value <= 1:
        return displacement, LD
    else:
        return displacement, LD ** (1 / p_value)
