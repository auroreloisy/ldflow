import numpy as np
from ldflow.flow import abc
from ldflow.section import make_section

PARAMETERS = {
    "section": "z",
    "value": 0,
    "Npoints": 200,  # number of point along each dimension
}
TAU = 20  # integration time
P_VALUE = 2  # Lp norm


def field(t, u):
    v = abc(t, u, parameters=(np.sqrt(3), np.sqrt(2), 1))
    return v


def run():

    make_section(
        field=field,
        parameters=PARAMETERS,
        tau=TAU,
        p_value=P_VALUE,
    )


if __name__ == "__main__":
    run()
