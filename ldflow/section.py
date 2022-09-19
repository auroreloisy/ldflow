import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from ldflow.ld import compute_lagrangian_descriptor


def _integrate(field, parameters):
    tau = parameters["tau"]
    p_value = parameters["p_value"]

    # forward integration
    d_forward, ld_forward = compute_lagrangian_descriptor(
        parameters={"section": parameters["section"], "value": parameters["value"], "n_points": parameters["n_points"]},
        vector_field=field,
        tau=tau,
        p_value=p_value,
    )

    # backward integration
    _, ld_backward = compute_lagrangian_descriptor(
        parameters={"section": parameters["section"], "value": parameters["value"], "n_points": parameters["n_points"]},
        vector_field=field,
        tau=-tau,
        p_value=p_value,
    )

    return d_forward, ld_forward, ld_backward


def _get_gradient_magnitude(ld):
    gradient_x, gradient_y = np.gradient(ld)
    g = np.sqrt(gradient_x**2 + gradient_y**2)
    return _normalize(g)


def _normalize(A):
    return (A - np.nanmin(A)) / (np.nanmax(A) - np.nanmin(A))


def _plot(displacement, ld_forward, ld_backward, parameters):
    # gradient
    ld_f_grad = _get_gradient_magnitude(ld_forward)
    ld_b_grad = -_get_gradient_magnitude(ld_backward)
    grad = ld_f_grad + ld_b_grad

    # scaling colors
    amp0 = np.max(np.abs(grad))
    amp1 = max([np.max(np.abs([displacement[i]])) for i in range(3)])

    levels = 100
    points = np.linspace(0, 2 * np.pi, parameters["n_points"])
    fig, ax = plt.subplots(2, 3)

    def _vmin(k):
        if k == 0 or k == 1:
            vmin = 0
        elif k == 2:
            vmin = - amp0
        else:
            vmin = -amp1
        return vmin

    def _vmax(k):
        if k == 0 or k == 1:
            vmax = 1
        elif k == 2:
            vmax = amp0
        else:
            vmax = amp1
        return vmax

    def _cmap(k):
        if k == 0:
            cmap = 'Purples_r'
        elif k == 1:
            cmap = 'Oranges_r'
        elif k == 2:
            cmap = 'PuOr'
        else:
            cmap = 'RdBu_r'
        return cmap

    # Lagrangian descriptor
    ax[0, 0].contourf(points, points, _normalize(ld_forward), cmap=_cmap(0), vmin=_vmin(0), vmax=_vmax(0), levels=levels)
    ax[0, 1].contourf(points, points, _normalize(ld_backward), cmap=_cmap(1), vmin=_vmin(1), vmax=_vmax(1), levels=levels)
    ax[0, 2].contourf(points, points, grad, cmap=_cmap(2), vmin=_vmin(2), vmax=_vmax(2), levels=levels)

    # Displacement
    ax[1, 0].contourf(points, points, displacement[0], cmap=_cmap(3), vmin=_vmin(3), vmax=_vmax(3), levels=levels)
    ax[1, 1].contourf(points, points, displacement[1], cmap=_cmap(4), vmin=_vmin(4), vmax=_vmax(4), levels=levels)
    ax[1, 2].contourf(points, points, displacement[2], cmap=_cmap(5), vmin=_vmin(5), vmax=_vmax(5), levels=levels)

    # Titles
    ax[0, 0].title.set_text("forward LD")
    ax[0, 1].title.set_text("backward LD")
    ax[0, 2].title.set_text("gradient magnitude")
    ax[1, 0].title.set_text("x-displacement")
    ax[1, 1].title.set_text("y-displacement")
    ax[1, 2].title.set_text("z-displacement")
    plt.annotate("flow = " + str(parameters["flow"])
                 + ", p = " + str(parameters["p_value"])
                 + ", tau = " + str(parameters["tau"])
                 + ", section " + parameters["section"] + " = " + str(parameters["value"]),
                 xy=(0.5, 0.96), xycoords="figure fraction", fontsize=12, ha='center'
                 )

    for i in range(2):
        for j in range(3):
            ax[i, j].set_aspect("equal", adjustable="box")
            ax[i, j].axis("off")
            k = 3*i + j
            sm = plt.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=_vmin(k), vmax=_vmax(k)), cmap=_cmap(k))
            plt.colorbar(sm, ax=ax[i, j])

    plt.show()


def make_section(field, parameters):

    d_forward, ld_forward, ld_backward = _integrate(field, parameters)
    _plot(d_forward, ld_forward, ld_backward, parameters)


