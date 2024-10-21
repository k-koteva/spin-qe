import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from typing import Union


def calculate_specific_power(temp: Union[int, float]) -> float:
    k = 2.86e5  # coefficient from the logarithmic fit
    return k * temp ** -2.09

def carnot_efficiency(temp: Union[int, float]) -> float:
    return (300 - temp) / temp
#     return 1 - (77 / temp)

print('Small system specific power at 1K: ', calculate_specific_power(1))
print('Carnot efficiency at 1K: ', carnot_efficiency(1))


def main():

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

    # Extract the magnitude (exponent) of k_fitted
    exponent = int(np.floor(np.log10(k_fitted)))
    mantissa = k_fitted / 10**exponent
    
    # Plot the data points and the fitted curve for the specific power function
    plt.figure(figsize=(10, 6))
    plt.scatter(temperatures_new, specific_power_new, color='red', label='Small systems', s=100)
    plt.plot(x_fit, y_fit, label=f'Fit: $y = {mantissa:.2f} \\cdot 10^{{{exponent}}} \\cdot x^{{-2.09}}$', color='blue', lw=2)

    # Plot the inverse Carnot COP curve
    plt.plot(x_fit, inverse_cop_fit, label='Carnot', color='green', lw=2)

    # Logarithmic scale for better visualization
    plt.xscale('log')
    plt.yscale('log')

    # Adjusting the font size of labels, title, and legend
    plt.xlabel('T cold [K]', fontsize=18)
    plt.ylabel('Specific power (W at 300 K)/(W at T cold)', fontsize=18)
    plt.title('Specific Power vs Cold Temperature for Cryogenic Refrigeration', fontsize=20)
    plt.legend(fontsize=16)

    # Adjust tick sizes
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Save the plot as a PDF
    plt.savefig('specific_power.pdf', format='pdf')

    # Optionally, close the plot to avoid displaying it
    plt.close()

    print(f"Fitted parameter k = {k_fitted:.2e}")

if __name__ == '__main__':
    main()

