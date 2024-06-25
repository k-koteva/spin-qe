from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from numpy.polynomial.polynomial import Polynomial
from pydantic import BaseModel

from spin_qe.device.spin_qubitsGlobal import SpinQubit

def fit_original_data(data_x: List[float], data_y: List[float]) -> None:
    print(f"Data x: {data_x}")
    print(f"Data y: {data_y}")

class DataPoint(BaseModel):
    x: float
    y: float

class DataArray(BaseModel):
    x: float
    y: List

def fit_polynomial(data_points: List[DataPoint], degree: int = 3) -> Polynomial:
    x_values = [point.x for point in data_points]
    y_values = [point.y for point in data_points]
    
    coefs = np.polynomial.polynomial.polyfit(x_values, y_values, degree)
    logger.info(f"coefs: {coefs}")
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

data_points_fidelity = [
    DataPoint(x=0.13595473785487275, y=100-99.88513496455832),
    DataPoint(x=0.3974260941764538, y=100-99.87824353928882),
    DataPoint(x=0.5983893453314294, y=100-99.86153839680675),
    DataPoint(x=0.7978210439802411, y=100-99.85089326537656),
    DataPoint(x=0.9976970611583134, y=100-99.80947907579504),
    DataPoint(x=1.0977326851818083, y=100-99.782012112168),
    DataPoint(x=1.199454699986816, y=100-99.73776258692122),
]

# Fit the polynomial with a degree of 3 (cubic) as a starting point
# polynomial = fit_polynomial(data_points, degree=3)

# Define a function to calculate the fitting function value for a given x
def calculate_fitting_function_value(x: float) -> float:
    return polynomial(x)

def plot_data_and_fit(data_points: List[DataPoint], polynomial: Polynomial):
    # Extract x and y values from data points
    x_values = [point.x for point in data_points]
    y_values = [point.y for point in data_points]
    
    # Generate a range of x values for plotting the fit
    x_fit = np.linspace(min(x_values), max(x_values), 500)
    y_fit = polynomial(x_fit)
    
    # Plot original data points
    plt.scatter(x_values, y_values, color='blue', label='Original Data')
    
    # Plot polynomial fit
    plt.plot(x_fit, y_fit, color='red', label='Polynomial Fit')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Original Data and Polynomial Fit')
    plt.legend()
    plt.show()

# Plot the data and the fitting curve
def plot_data(data_points: List[DataPoint], simulation_data: List[DataPoint]):
    # Extract x and y values from data points
    x_values_data = [point.x for point in data_points]
    y_values_data = [point.y for point in data_points]
    
    x_values_simulation = [point.x for point in simulation_data]
    y_values_simulation = [point.y for point in simulation_data]
    
    
    # Plot original data points
    plt.scatter(x_values_data, y_values_data, color='blue', label='Experimental Data')
    plt.scatter(x_values_simulation, y_values_simulation, color='green', label='Model Predictions')
    # plt.ylim(99.72, 100)
    
    plt.xlabel('Temperature of the qubit (K)')
    plt.ylabel('Single Qubit Fidelity (%)')
    plt.title('Fidelity of a spin qubit at 1.44 MHz Rabi frequency')
    plt.legend()
    plt.savefig('fidelity_update.pdf', format='pdf')
    plt.show()

def plot_more_data(data_points: List[DataPoint], simulation_data: List[DataPoint], simulation_data_noisy: List[DataPoint]):
    # Extract x and y values from data points
    x_values_data = [point.x for point in data_points]
    y_values_data = [point.y for point in data_points]
    
    x_values_simulation = [point.x for point in simulation_data]
    y_values_simulation = [point.y for point in simulation_data]

    x_values_simulation_noisy = [point.x for point in simulation_data_noisy]
    y_values_simulation_noisy = [point.y for point in simulation_data_noisy]
    
    
    # Plot original data points
    plt.scatter(x_values_data, y_values_data, color='blue', label='Experimental Data')
    plt.scatter(x_values_simulation_noisy, y_values_simulation_noisy, color='red', label='Noisy Model Predictions')
    plt.scatter(x_values_simulation, y_values_simulation, color='green', label='Model Predictions')
    # plt.ylim(99.72, 100)
    
    plt.xlabel('Temperature of the qubit (K)')
    plt.ylabel('Single Qubit Fidelity (%)')
    plt.title('Fidelity of a spin qubit at 1.44 MHz Rabi frequency')
    plt.legend()
    plt.savefig('fidelity_update.pdf', format='pdf')
    plt.show()

def plot_noisy_data(data_points: List[DataPoint], simulation_data: List[DataPoint], simulation_dataarray_noisy: List[DataArray]):
    # Extract x and y values from data points
    x_values_data = [point.x for point in data_points]
    y_values_data = [point.y for point in data_points]
    
    x_values_simulation = [point.x for point in simulation_data]
    y_values_simulation = [point.y for point in simulation_data]
    
    for point in simulation_dataarray_noisy:
        x_values_simulation_noisy = point.x
        print(f'x_value: {x_values_simulation_noisy}')
        y_values_simulation_noisy = point.y
        print(f'y_value: {y_values_simulation_noisy}')
        for y_value in y_values_simulation_noisy:
            plt.scatter(x_values_simulation_noisy, 100-y_value, color='red')
        plt.scatter(x_values_simulation_noisy, 100-np.mean(y_values_simulation_noisy), color='black')
    
    # Plot original data points
    plt.scatter(x_values_data, y_values_data, color='blue', label='Experimental Data')
    plt.scatter(x_values_simulation, y_values_simulation, color='green', label='Model Predictions')
    # plt.ylim(99.72, 100)
    plt.yscale('log')
    
    plt.xlabel('Temperature of the qubit (K)')
    plt.ylabel('Single Qubit Fidelity (%)')
    plt.title('Fidelity of a spin qubit at 1.44 MHz Rabi frequency')
    plt.legend()
    plt.savefig('fidelity_update.pdf', format='pdf')
    plt.show()


    # New data points for the specified function
new_data_points_function = [
    DataPoint(x=0.14059971957346504, y=3.50167594550633),
    DataPoint(x=0.2005094879453901, y=3.289615102239407),
    DataPoint(x=0.2998049490932273, y=3.0181552789394224),
    DataPoint(x=0.4001423479529536, y=2.8124981079872016),
    DataPoint(x=0.49982212835796797, y=2.68345172421436),
    DataPoint(x=0.5982991502664093, y=2.5197746911913907),
    DataPoint(x=0.6994305446370607, y=2.3847605706780395),
    DataPoint(x=0.7985352726495942, y=2.329539227371849),
    DataPoint(x=0.8988300750079263, y=2.367503681395887),
    DataPoint(x=0.9974590332327369, y=2.276138045498526),
    DataPoint(x=1.0964828938581461, y=2.170967470250422),
    DataPoint(x=1.1939825352856186, y=2.1373050943870338),
    DataPoint(x=1.2940133244042942, y=2.0875262575507483),
    DataPoint(x=1.3958031773404724, y=2.0879445319434398),
]

# Re-fit the polynomial with the new data set for this function, assuming the same degree (3) is appropriate


def main():
    # Original data points for the specified function
    x_values = [0.13595473785487275, 0.3974260941764538, 0.5983893453314294, 0.7978210439802411, 0.9976970611583134, 1.0977326851818083, 1.199454699986816]
    y_values = [0.9988513496455832, 0.9987824353928882, 0.9986153839680675, 0.9985089326537656, 0.9980947907579504, 0.99782012112168, 0.9973776258692122]
    fit_original_data(data_x=x_values, data_y=y_values)
    # plot_data_and_fit(data_points, polynomial)

    # fidelities = []
    # sim_data = []
    # noisy_sim_data = []
    # noisy_sim_dataarray = []
    # temperatures = np.linspace(0.01, 1.4, 10)
    # for temp in temperatures:
    #     spin_qubit_params = {
    #         'n_q': 1,
    #         'Tq': temp,
    #         'f': 11.20,
    #         'f_in_GHz':11.20e9,
    #         'rabi_in_MHz': 1.44e6,
    #         'rabi': 1.44,
    #         'atts_list': [3, 0],
    #         'stages_ts': [4, 300],
    #         'silicon_abs': 0.0
    #     }
    #     spin_qubit = SpinQubit(**spin_qubit_params)
        # fidelities.append(spin_qubit.fid_1q())

        # sim_data.append(DataPoint(x=temp, y=spin_qubit.T2HQ()))
        # sim_data.append(DataPoint(x=temp, y=spin_qubit.T2Q()))
        # sim_data.append(DataPoint(x=temp, y=100-spin_qubit.fid_1q()*100))
        # noisy_sim_data.append(DataPoint(x=temp, y=spin_qubit.noiseFid()*100))
        # noisy_sim_dataarray.append(DataArray(x=temp, y=spin_qubit.noiseFidarray()))


        # def fid_1q(T2) -> float:
        #     T2 = spin_qubit.T2Q()
        #     rabi = spin_qubit.rabi
        #     print(f"rabi: {rabi}")
        #     logger.info(f"rabi: {rabi}")
        #     delta = 1/T2
        #     prob = 1 - (np.pi+1)*delta**2/(4*rabi**2)
        #     return prob
        # my_fid_1q = fid_1q(spin_qubit.T2Q())
        # logger.warning(f"FIDELITY: {my_fid_1q}")

    # print(sim_data)
    # plot_data(data_points, sim_data)
    # plot_data(data_points_fidelity, sim_data)
    # plot_more_data(data_points_fidelity, sim_data, noisy_sim_data)
    # plot_noisy_data(data_points_fidelity, sim_data, noisy_sim_dataarray)
    # logger.info(spin_qubit.T2HQ())
    # logger.info(spin_qubit.T2Q())

    # new_polynomial_function = fit_polynomial(new_data_points_function, degree=3)

    # # Plot the new data and the new fitting curve for this function
    # plot_data_and_fit(new_data_points_function, new_polynomial_function)

    # # Calculate the value of the fitting function at a specific x value


if __name__ == "__main__":
    main()
