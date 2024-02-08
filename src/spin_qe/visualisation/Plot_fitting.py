from typing import List, Tuple

import numpy as np
from loguru import logger
from numpy.polynomial.polynomial import Polynomial
from pydantic import BaseModel


class DataPoint(BaseModel):
    x: float
    y: float

def fit_polynomial(data_points: List[DataPoint], degree: int = 3) -> Polynomial:
    x_values = [point.x for point in data_points]
    y_values = [point.y for point in data_points]
    
    coefs = np.polynomial.polynomial.polyfit(x_values, y_values, degree)
    print(coefs)
    poly = Polynomial(coefs)
    
    return poly

data_points = [
    DataPoint(x=0.14075165233102482, y=95.43414982614362),
    DataPoint(x=0.20000032995449005, y=93.27459565407987),
    DataPoint(x=0.30085010869294715, y=84.21816216491546),
    DataPoint(x=0.4019048790647898, y=73.05839816700546),
    DataPoint(x=0.5023777835866758, y=61.874712990032116),
    DataPoint(x=0.601701210054336, y=51.98269484553438),
    DataPoint(x=0.7037559395788873, y=45.437919539133816),
    DataPoint(x=0.8000039594604119, y=40.6714939397868),
    DataPoint(x=0.9051078790055805, y=37.28216935625001),
    DataPoint(x=1.0047572247914054, y=32.843466297729705),
    DataPoint(x=1.104837029550485, y=29.866131401870614),
    DataPoint(x=1.2091312655086772, y=27.593071335830338),
    DataPoint(x=1.3107654109338298, y=24.695457882720838),
    DataPoint(x=1.4075142012320956, y=22.27775730379143),
]

# Fit the polynomial with a degree of 3 (cubic) as a starting point
polynomial = fit_polynomial(data_points, degree=3)

# Define a function to calculate the fitting function value for a given x
def calculate_fitting_function_value(x: float) -> float:
    return polynomial(x)

# Example calculation for x = 0.5
example_x = 0.5
example_y = calculate_fitting_function_value(example_x)
logger.info(f"Example calculation for x = {example_x}: y = {example_y}") 