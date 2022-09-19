import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ldflow.ld import compute_lagrangian_descriptor


def _integrate(field, parameters):
    n_points = parameters["n_points"]
    tau = parameters["tau"]
    p_value = parameters["p_value"]

    # forward integration
    _, x_plane = compute_lagrangian_descriptor(
        parameters={"section": "x", "value": 0, "n_points": n_points},
        vector_field=field,
        tau=tau,
        p_value=p_value,
    )
    _, y_plane = compute_lagrangian_descriptor(
        parameters={"section": "y", "value": 0, "n_points": n_points},
        vector_field=field,
        tau=tau,
        p_value=p_value,
    )
    _, z_plane = compute_lagrangian_descriptor(
        parameters={"section": "z", "value": 0, "n_points": n_points},
        vector_field=field,
        tau=tau,
        p_value=p_value,
    )

    return x_plane, y_plane, z_plane


def _transform(planes, show_gradient, power):
    if show_gradient:
        def _get_gradient_magnitude(ld):
            gradient_x, gradient_y = np.gradient(ld)
            g = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
            g = _normalize(g)
            g = g ** power  # tweak to make features stand out
            return g

        def _normalize(A):
            return (A - np.nanmin(A)) / (np.nanmax(A) - np.nanmin(A))

        new_planes = []
        for plane in planes:
            new_planes.append(_get_gradient_magnitude(plane))
        return new_planes
    else:
        return planes


def _plot(planes, parameters):
    n_points = parameters["n_points"]
    colormap = parameters["colormap"]

    vmin = min([np.min(planes[i]) for i in range(3)])
    vmax = max([np.max(planes[i]) for i in range(3)])

    levels = 100
    points = np.linspace(0, 2 * np.pi, n_points)

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    X = points
    Y = points
    X, Y = np.meshgrid(X, Y)

    cset = [[], [], []]

    # z-plane
    cset[0] = ax.contourf(X, Y, planes[2], zdir='z', offset=2*np.pi,
                          levels=levels, vmin=vmin, vmax=vmax, cmap=colormap)

    # x-plane
    cset[1] = ax.contourf(planes[0], X, Y, zdir='x', offset=2*np.pi,
                          levels=levels, vmin=vmin, vmax=vmax, cmap=colormap)

    # y-plane
    cset[2] = ax.contourf(X, planes[1], Y, zdir='y', offset=0,
                          levels=levels, vmin=vmin, vmax=vmax, cmap=colormap)

    # edges
    line_color = "Grey"
    line_width = 1
    x0 = 0 - 0.01
    x1 = 2 * np.pi + 0.01
    ax.plot(
        [x1, x1, x1, x1, x1, x0, x0, x1],
        [x0, x1, x1, x0, x0, x0, x0, x0],
        [x0, x0, x1, x1, x0, x0, x1, x1],
        color=line_color,
        linewidth=line_width,
        zorder=1e6,
    )
    ax.plot(
        [x0, x0, x1],
        [x0, x1, x1],
        [x1, x1, x1],
        color=line_color,
        linewidth=line_width,
        zorder=1e6,
    )

    # setting 3D-axis-limits:
    ax.set_xlim3d(0, 2*np.pi)
    ax.set_ylim3d(0, 2*np.pi)
    ax.set_zlim3d(0, 1.7*np.pi)  # tweak for adjusting aspect ratio

    # ax.set_xlabel("x")
    # ax.set_ylabel("y")
    # ax.set_zlabel("z")

    # Hide grid lines
    plt.axis('off')

    # Titles
    plt.annotate("flow = " + str(parameters["flow"])
                 + ", p = " + str(parameters["p_value"])
                 + ", tau = " + str(parameters["tau"]),
                 xy=(0.5, 0.96), xycoords="figure fraction", fontsize=12, ha='center'
                 )

    plt.show()


def make_cube(field, parameters):

    planes = _integrate(field, parameters)
    planes = _transform(planes, parameters["gradient"], parameters["gradient_power"])
    _plot(planes, parameters)


