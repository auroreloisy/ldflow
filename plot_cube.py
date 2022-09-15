import numpy as np
from ldflow.cube import make_cube
from ldflow.flow import abc


N_POINTS = 200  # number of point along each dimension
TAU = 20  # integration time
P_VALUE = 2  # Lp norm
GRADIENT = True  # plot gradient of LD instead of LD
COLORMAP = 'afmhot'  # "afmhot", "bone", "bone_r"


def field(t, u):
    v = abc(t, u, parameters=(np.sqrt(3), np.sqrt(2), 1))
    return v


def run():
    make_cube(
        field=field,
        n_points=N_POINTS,
        tau=TAU,
        p_value=P_VALUE,
        show_gradient=GRADIENT,
        colormap=COLORMAP,
    )


if __name__ == "__main__":
    run()
