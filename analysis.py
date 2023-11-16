import numpy as np

# temporary store of variables
freq = 5413.43
freq_error = 338.845
wave_length = 632.8e-9
D = 3.6e-2
D_error = 0.2e-2
focal = 1.3e-1
R = 5e-3
R_error = 0.1e-3


# functions
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


def calculate_flow_speed(freq, wave_length, D, focal):
    return (freq * wave_length * np.sqrt((D / 2) ** 2 + focal**2)) / D


# loose example calculation
flow_speed = calculate_flow_speed(freq, wave_length, D, focal)
flow_speed_error = error_prop_flow_speed(
    wave_length, freq, freq_error, D, D_error, focal
)

print(f"Flow speed: {flow_speed}m/s\nFlow speed error: +/-{flow_speed_error}m/s")
