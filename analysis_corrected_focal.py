import numpy as np
from os import getcwd, mkdir, path
import matplotlib.pyplot as plt

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

def calculate_sin(A,B):
    return A / np.sqrt(A**2 + B**2) 

def error_prop_sin_2(A,A_error,B,B_error):
    return np.sqrt( ((A**2+B**2)**(-0.5) * (1 - A**2*(A**2+B**2)**-1 * A_error))**2 + (-A*B*(A**2+B**2)**(-3/2)*B_error) )


# functions to propogate the errors
def error_prop_sin(D, focal, D_error):
    return (
        ((D / 2) ** 2 + focal**2) ** (-0.5)
        * (1 - ((D / 2) ** 2 + focal**2) ** -1)
        * D_error
    )


def error_prop_flow_speed(wave_length, freq, freq_error, D, D_error, focal):
    return np.sqrt(
        ((freq_error * wave_length * np.sqrt((D / 2) ** 2 + (focal) ** 2)) / (D)) ** 2
        + (
            ((-freq * wave_length * 2 * ((D / 2) ** 2 + (focal) ** 2)) / (D) ** 2)
            * error_prop_sin(D, focal, D_error)
        )
        ** 2
    )


# function for calculating the flowspeed
def calculate_flow_speed(freq, wave_length, D, focal):
    return (freq * wave_length * np.sqrt((D / 2) ** 2 + focal**2)) / D


# loose example calculation
flow_speed = calculate_flow_speed(freq, wave_length, D, focal)
flow_speed_error = error_prop_flow_speed(
    wave_length, freq, freq_error, D, D_error, focal
)

# showing calculation
font_size = 14
title_size = 17

print(f"Flow speed: {flow_speed}m/s\nFlow speed error: +/-{flow_speed_error}m/s")
plt.errorbar(
    R, flow_speed, yerr=flow_speed_error, xerr=R_error, linestyle="None", marker="o"
)
plt.ylabel("flow speed (m/s)")
plt.xlabel("radial distance(m)")
plt.title("flow speed vs radial distance")
plt.xticks(rotation=-45)
plt.tight_layout()
plt.savefig(plot_path + plot_name)
plt.show()
