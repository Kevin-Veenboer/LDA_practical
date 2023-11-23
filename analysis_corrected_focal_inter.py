import numpy as np
from os import getcwd, mkdir, path
import matplotlib.pyplot as plt
import pandas as pd
from lmfit import Model

# might add non-overwerite feature
plot_path = f"{getcwd()}/Plots/"
if not path.exists(plot_path):
    mkdir(plot_path)
    plot_name = "ExamplePlot_0.jpg"
else:
    plot_name = "ExamplePlot_0.jpg"


def convert_to_sigma(FWHM):
    return FWHM / (2 * np.sqrt(2 * np.log(2)))


def calculate_sin(A, B):
    return A / np.sqrt(A**2 + B**2)


# functions to propogate the errors


def error_prop_sin(A, A_error, B, B_error):
    return np.sqrt(
        (
            ((A**2 + B**2) * -0.5)
            * (1 - (A**2) * (A**2 + B**2) ** -1)
            * A_error
        )
        ** 2
        + (-A * B * (A**2 + B**2) ** (-3 / 2) * B_error) ** 2
    )


def error_prop_flow_speed(wave_length, freq, freq_error, A, A_error, B, B_error):
    return np.sqrt(
        ((wave_length * freq_error) / (2 * calculate_sin(A, B))) ** 2
        + (
            (-freq * wave_length * error_prop_sin(A, A_error, B, B_error))
            / (2 * calculate_sin(A, B) ** 2)
        )
        ** 2
    )


# function for calculating the flowspeed
def calculate_flow_speed(freq, wave_length, A, B):
    return (freq * wave_length) / (2 * calculate_sin(A, B))


def fit_func(R, C, Middle, Tube_radius):
    return C * (1 - ((R - Middle) ** 2 / Tube_radius**2))


def test_func(R, C=0.008, Middle=0.08, Tube_radius=0.126):
    return C * (1 - ((R - Middle) ** 2 / Tube_radius**2))


def correct_radial(DF):
    radials = DF["R"]
    freqs = DF["Freq"]
    comb = [(R, V) for R, V in zip(radials, freqs)]
    comb.sort(key=lambda x: x[1])
    middle = comb[-1][0]
    DF["R"] = DF["R"] - middle
    return DF


# LOAD CSV
data_path = f"{getcwd()}/data_17_11.csv "
data = pd.read_csv(data_path)

wave_length = 632e-9

A = 2.67e-3
A_error = 1e-3
B = 10e-3
B_error = 0.1e-3

data["Freq_error"] = convert_to_sigma(data["FWHM"])

R = data["R"] * 1e-3
R_error = data["R_error"] * 1e-3
freq = data["Freq"]
freq_error = data["Freq_error"]

Flow_speeds = calculate_flow_speed(freq, wave_length, A, B)
Flow_speeds_error = error_prop_flow_speed(
    wave_length, freq, freq_error, A, A_error, B, B_error
)


"""
SHOWING RESULTS
"""


model = Model(fit_func)
params = model.make_params()
params["C"].set(value=0.001)
params["Middle"].set(value=0.035, min=0.001, max=0.01)
params["Tube_radius"].set(value=0.011)

results = model.fit(Flow_speeds, weights=Flow_speeds_error, R=R, params=params)
print(results.fit_report())
with open(f"{getcwd()}/Fit_Results/last_fit_report.txt", "w") as file:
    print(results.fit_report(), file=file)

plt.errorbar(
    R, Flow_speeds, yerr=Flow_speeds_error, xerr=R_error, linestyle="None", marker="o"
)
plt.plot(R, results.best_fit, "-", label="best fit")
plt.legend()
plt.xlim(0, 0.0125)
plt.ylim(0, 0.01)
plt.ylabel("flow speed (m/s)")
plt.xlabel("radial distance(m)")
plt.title("flow speed vs radial distance")
plt.xticks(rotation=-45)
plt.tight_layout()
plt.savefig(f"{getcwd()}/Fit_Results/last_fit_plot.svg")
plt.savefig(f"{getcwd()}/Fit_Results/last_fit_plot.jpg")
plt.show()
