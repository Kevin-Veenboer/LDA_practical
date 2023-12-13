import numpy as np
from os import getcwd, mkdir, path, listdir
import matplotlib.pyplot as plt
import pandas as pd
from lmfit import Model


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


# General variables
wave_length = 632e-9
A = 2.67e-3
A_error = 1e-3
B = 10e-3
B_error = 0.1e-3


plot_colors = ["gray", "brown", "cyan", "purple", "blue", "red", "green", "orange"]
IDs_to_plot = ["4", "6", "8", "10"]
C_to_skip = [4, 6]

# IDs_to_plot = ["4", "8", "12", "16", "18", "20", "22", "24"]
# C_to_skip = []

data_dict = {}
data_path = f"{getcwd()}/DataExtensive_8_12/"
save_path = f"{getcwd()}/ExtendedResults/"

for data_file in listdir(data_path):
    # get file path and ID, then load it as DF
    file_path = data_path + data_file
    file_ID = data_file[5:-4]
    data = pd.read_csv(file_path, sep=";")

    # Convert some data to correct units
    data["Freq_error"] = convert_to_sigma(data["FWHM"])
    R = data["R"] * 1e-3
    R_error = data["R_error"] * 1e-3
    freq = data["Freq"]
    freq_error = data["Freq_error"]

    # Calculate flow speeds and errors
    Flow_speeds = calculate_flow_speed(freq, wave_length, A, B)
    Flow_speeds_error = error_prop_flow_speed(
        wave_length, freq, freq_error, A, A_error, B, B_error
    )

    # Define model and set parameters
    model = Model(fit_func)
    params = model.make_params()
    params["C"].set(value=0.001)
    params["Middle"].set(value=0.013, min=0.008, max=0.018)
    params["Tube_radius"].set(value=0.011, min=0.001, max=0.02)

    # Fit the model
    results = model.fit(Flow_speeds, weights=Flow_speeds_error, R=R, params=params)

    # get the value for C and store it in the general data dict
    data_dict[file_ID] = (results.params["C"].value, results.params["C"].stderr)

    if file_ID in IDs_to_plot:
        color = plot_colors.pop()
        plt.errorbar(
            R,
            Flow_speeds,
            yerr=Flow_speeds_error,
            xerr=R_error,
            linestyle="None",
            marker="o",
            label=f"{file_ID}cm",
            color=color,
        )
        plt.plot(
            R,
            results.best_fit,
            linestyle="dashed",
            label=f"fit: {file_ID}cm",
            color=color,
            alpha=0.7,
        )

plt.legend(loc="upper right")
plt.xlim(0.006, 0.022)
plt.ylim(0.000, 0.016)
plt.ylabel("flow speed (m/s)")
plt.xlabel("radial distance(m)")
plt.title("flow speed vs radial distance")
plt.xticks(rotation=-45)
plt.tight_layout()
plt.savefig(f"{save_path}MultiPlot.jpg", dpi=300)
plt.savefig(f"{save_path}MultiPlot.svg")
plt.show()

# Extract the C parameter data into a list and sort on ID
C_data = sorted(
    [(int(ID), values) for ID, values in data_dict.items()], key=lambda x: x[0]
)

# Split data up into seperate lists to print
lengths = []
C_values = []
C_errors = []

for element in C_data:
    if element[0] in C_to_skip:
        continue
    lengths.append(element[0])
    C_values.append(element[1][0])
    C_errors.append(element[1][1])

plt.errorbar(lengths, C_values, yerr=C_errors, linestyle="None", marker="o")
plt.ylabel("Maximum of fitted flow speed (m/s)")
plt.xlabel("Distance from tube entrance (cm)")
plt.title("maximum flow speed vs entrance distance")
plt.xlim(0, 25)
plt.ylim(0.0055, 0.0145)
plt.tight_layout()
plt.savefig(f"{save_path}C_plot.jpg", dpi=300)
plt.savefig(f"{save_path}C_plot.svg")
plt.show()

print(C_data)
