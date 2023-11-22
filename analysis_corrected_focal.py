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

# temporary store of variables
freq = 5413.43
freq_error = 338.845
wave_length = 632.8e-9
D = 3.6e-2
D_error = 0.2e-2
focal = 1.3e-1
R = 5e-3
R_error = 0.1e-3


def calculate_sin(A, B):
    return A / np.sqrt(A**2 + B**2)


# functions to propogate the errors


def error_prop_sin(A, A_error, B, B_error):
    return np.sqrt(
        ((A**2 + B**2) ** (-0.5) * (1 - A**2 * (A**2 + B**2) ** -1 * A_error))
        ** 2
        + (-A * B * (A**2 + B**2) ** (-3 / 2) * B_error)
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


def fit_func(R, C, Tube_radius):
    return C * (1 - (R**2 / Tube_radius**2))


# LOAD CSV
data_path = f"{getcwd()}/data_17_11.csv "
data = pd.read_csv(data_path)

wave_length = 632e-9

A = 2.67e-3
A_error = 2e-3
B = 10e-3
B_error = 0.1e-3

R = data["R"] * 1e-3
R_error = data["R_error"] * 1e-3
freq = data["Freq"]
freq_error = data["Freq_error"]

Flow_speeds = calculate_flow_speed(freq, wave_length, A, B)
Flow_speeds_error = error_prop_flow_speed(
    wave_length, freq, freq_error, A, A_error, B, B_error
)


model = Model(fit_func)

results = model.fit(Flow_speeds, R=R, C=1, Tube_radius=0.02)
print(results.fit_report())

plt.plot(R, Flow_speeds, "o")
plt.plot(R, results.best_fit, "-", label="best fit")
plt.legend()
plt.show()


# #printing
# plt.errorbar(
#     R, Flow_speeds, yerr=Flow_speeds_error, xerr=R_error, linestyle="None", marker="o"
# )
# # plt.errorbar(R, Flow_speeds, xerr=R_error, linestyle="None", marker="o")
# plt.ylabel("flow speed (m/s)")
# plt.xlabel("radial distance(m)")
# plt.title("flow speed vs radial distance")
# plt.xticks(rotation=-45)
# plt.tight_layout()
# plt.show()
