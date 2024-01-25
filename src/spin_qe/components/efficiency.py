# import numpy as np
# from scipy.optimize import curve_fit

# # Data points
# x = np.array([93.13479631662334, 0.011945933071178236, 1.5231903609728097])
# y = np.array([17.7615249191661, 2507081193.8774233, 98103.38476689655])

# # Power law function
# def power_law(x, k):
#     return k * x ** -2.1

# # Fit the data to the power law function
# popt, _ = curve_fit(power_law, x, y)

# # The value of the k coefficient
# k_coefficient = popt[0]
# print(k_coefficient)


# from scipy.stats import linregress

# # Logarithms of x and y
# log_x = np.log(x)
# log_y = np.log(y)

# # Linear regression on the logarithmic values
# slope, intercept, r_value, p_value, std_err = linregress(log_x, log_y)

# # Calculate the k coefficient from the intercept of the linear fit
# k_coefficient_log = np.exp(intercept)
# k_coefficient_log, slope, r_value**2 # returning the slope and r-squared for reference

from typing import Union


# Power law function definition
# Define the power law function using the determined coefficient k
def calculate_specific_power(temp: Union[int, float]) -> float:
    k = 236236.0  # coefficient from the logarithmic fit
    return k * temp ** -2.1



def power_law(x):
    return 236,236 * x ** -2.1

print('Power at 0.01: ', calculate_specific_power(0.011945933071178236))
print('Power at 1.5: ', calculate_specific_power(1.5231903609728097))
print('Power at 93: ', calculate_specific_power(93.13479631662334))

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# Given data points
x = np.array([93.13479631662334, 0.011945933071178236, 1.5231903609728097])
y = np.array([17.7615249191661, 2507081193.8774233, 98103.38476689655])


# Generate a range of x values for plotting the fitted line
x_values_for_plot = np.logspace(np.log10(min(x)), np.log10(max(x)), 100)

# Calculate the y values using the power law function
y_values_for_plot = calculate_specific_power(x_values_for_plot)

# Create the plot
plt.figure(figsize=(10, 6))

# Plot the original data points
plt.scatter(x, y, color='blue', label='Data points')

# Plot the power law fit line
plt.plot(x_values_for_plot, y_values_for_plot, color='red', label='Fit $y = 236236 \cdot x^{-2.1}$')

# Set the x and y axis to logarithmic scale
plt.xscale('log')
plt.yscale('log')

# Labeling the axes and the title
plt.xlabel('T cold [K]')
plt.ylabel('Specific power [W/(W/K)]')
plt.title('Specific Power as a Function of Cold Temperature')

# Add a legend
plt.legend()

# Show the plot
plt.show()
