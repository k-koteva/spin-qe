import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# New data points (temperatures in Kelvin)
temperatures_new = np.array([75.2752111463307, 38.27949732694324, 19.204665835694286, 
                             3.9999613993340564, 0.09569966683507591, 0.01913984101502063])

# Corresponding specific power values
specific_power_new = np.array([42.169650342858226, 103.6632928437698, 453.1583637600818, 
                               8058.421877614818, 32781211.513934586, 1113973859.9948025])

# Define the function to fit: y = k * temp ** -2.09
def specific_power_function(temp, k):
    return k * temp ** -2.09

# Fit the curve to the new data points
params, covariance = curve_fit(specific_power_function, temperatures_new, specific_power_new)

# Extract the fitted value of k
k_fitted = params[0]

# Generate a smooth line for the fitted curve
x_fit = np.linspace(min(temperatures_new), max(temperatures_new), 1000)
y_fit = specific_power_function(x_fit, k_fitted)

# Define the Carnot COP and its inverse
def inverse_carnot_cop(T_cold, T_hot=300):
    return (T_hot - T_cold) / T_cold

# Calculate the inverse Carnot COP for the fitted temperature range
inverse_cop_fit = inverse_carnot_cop(x_fit)

# Plot the data points and the fitted curve for the specific power function
plt.figure(figsize=(10, 6))
plt.scatter(temperatures_new, specific_power_new, color='red', label='Specific Power Data Points', s=100)
plt.plot(x_fit, y_fit, label=f'Fit: $y = {k_fitted:.2e} \cdot x^{{-2.09}}$', color='blue', lw=2)

# Plot the inverse Carnot COP curve
plt.plot(x_fit, inverse_cop_fit, label='Inverse Carnot COP', color='green', lw=2)

# Logarithmic scale for better visualization
plt.xscale('log')
plt.yscale('log')

# Adjusting the font size of labels, title, and legend
plt.xlabel('Temperature (Kelvin)', fontsize=18)
plt.ylabel('Value', fontsize=18)
plt.title('Specific Power Function vs Inverse Carnot COP', fontsize=20)
plt.legend(fontsize=16)

# Adjust tick sizes
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Save the plot as a PDF
plt.savefig('specific_power_vs_inverse_carnot_cop.pdf', format='pdf')

# Optionally, close the plot to avoid displaying it
plt.close()

print(f"Fitted parameter k = {k_fitted:.2e}")