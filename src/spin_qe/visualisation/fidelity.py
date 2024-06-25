import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from pydantic import BaseModel
import pandas as pd

# Define DataPoint model
class DataPoint(BaseModel):
    x: float
    y: float

# Given data points
data_points_fidelity = [
    DataPoint(x=0.13595473785487275, y=0.9988513496455832),
    DataPoint(x=0.3974260941764538, y=0.9987824353928882),
    DataPoint(x=0.5983893453314294, y=0.9986153839680675),
    DataPoint(x=0.7978210439802411, y=0.9985089326537656),
    DataPoint(x=0.9976970611583134, y=0.9980947907579504),
    DataPoint(x=1.0977326851818083, y=0.99782012112168),
    DataPoint(x=1.199454699986816, y=0.9973776258692122),
]

# Extract x and y values from the data points
x_data = np.array([point.x for point in data_points_fidelity])
y_data = np.array([point.y for point in data_points_fidelity])

# Define the model function
def model_function(x, m, b, c):
    f = 1.44e6
    return 1 - m * ((b + c * x)**2) / (f**4)

# Fit the curve
popt, pcov = curve_fit(model_function, x_data, y_data)

# Extract the fitting parameters
m, b, c = popt

print(f"Fitted parameters: m = {m}, b = {b}, c = {c}")

# Define the function y(x) using the fitted parameters
def y_function(x):
    return 1 - m * ((b + c * x**2)**2) / (1.44e6**4)

# Generate y values for the fitted curve
x_fit = np.linspace(min(x_data), max(x_data), 100)
y_fit = y_function(x_fit)

# Plot the data points and the fitted curve
plt.figure(figsize=(10, 6))
plt.scatter(x_data, y_data, color='red', label='Data Points')
plt.plot(x_fit, y_fit, color='blue', label='Fitted Function')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Data Points and Fitted Function')
plt.legend()
plt.grid(True)
plt.show()

# Prepare data for display
fitted_data = pd.DataFrame({"x": x_data, "y": y_data, "fitted_y": y_function(x_data)})

# import ace_tools as tools; tools.display_dataframe_to_user(name="Fitted Parameters and Data Points", dataframe=fitted_data)

# from pydantic import BaseModel
# from typing import List

# class DataPoint(BaseModel):
#     x: float
#     y: float

# data_points_fidelity = [
#     DataPoint(x=0.13595473785487275, y=0.9988513496455832),
#     DataPoint(x=0.3974260941764538, y=0.9987824353928882),
#     DataPoint(x=0.5983893453314294, y=0.9986153839680675),
#     DataPoint(x=0.7978210439802411, y=0.9985089326537656),
#     DataPoint(x=0.9976970611583134, y=0.9980947907579504),
#     DataPoint(x=1.0977326851818083, y=0.99782012112168),
#     DataPoint(x=1.199454699986816, y=0.9973776258692122),
# ]

# inverted_data_points = [
#     DataPoint(x=dp.x, y=1-dp.y) for dp in data_points_fidelity
# ]

# print(inverted_data_points)