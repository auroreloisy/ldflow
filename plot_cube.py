"""Plot LD on the faces of a cube to highlight flow structures"""
import numpy as np
from ldflow.cube import make_cube
from ldflow.flow import abc, tgv_2d, turb_frozen

PARAMETERS = {
    "flow": "abc",  # "abc", "tgv_2d", "turb_frozen_3d"
    "n_points": 200,  # number of point along each dimension
    "tau": 20,  # integration time
    "p_value": 2,  # Lp norm
    "gradient": True,  # plot gradient of LD instead of LD
    "gradient_power": 0.75,  # tweak to make features stand out
    "colormap": 'afmhot',  # "afmhot", "bone"
    "show_colorbar": True,  # whether to show the colorbar on the plot
}

if PARAMETERS["flow"] == "abc":

    field = abc(a=np.sqrt(3), b=np.sqrt(2), c=1.0)

elif PARAMETERS["flow"] == "tgv_2d":

    field = tgv_2d()

elif PARAMETERS["flow"] == "turb_frozen_3d":

    # field = turb_frozen(n_grid=128, file='v000121.csv', print_info=True)
    # field = turb_frozen(n_grid=128, file='v006018.csv', print_info=True)
    field = turb_frozen(n_grid=128, file='v014863.csv', print_info=True)

else:
    raise Exception("This flow is not implemented")


def run():
    make_cube(
        field=field,
        parameters=PARAMETERS,
    )


if __name__ == "__main__":
    run()
