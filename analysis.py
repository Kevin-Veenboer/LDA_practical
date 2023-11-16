import numpy as np


def error_prop_sin(D, Foc, D_error):
    return (D / 2) ** 2 + Foc**2
