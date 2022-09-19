import numpy as np
from ldflow.flow import abc, turb_frozen, tgv_2d
from ldflow.section import make_section

PARAMETERS = {
    "flow": "abc",  # "abc", tgv_2d, "turb_frozen_3d"
    "section": "z",
    "value": 0,
    "n_points": 200,  # number of point along each dimension
    "tau": 20,  # integration time
    "p_value": 2  # Lp norm
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

    make_section(
        field=field,
        parameters=PARAMETERS,
    )


if __name__ == "__main__":
    run()
