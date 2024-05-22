from math import exp
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from pydantic import BaseModel, Field, confloat, conint, validator
from scipy.constants import h, k

from spin_qe.device.spin_qubitsGlobal import SpinQubit


# Assuming Cryo and SpinQubit classes are defined as per your provided code
# Define the StageData class to hold our data
class StageData(BaseModel):
    temperature: float
    n_q: int
    cables_power: float
    cryo_power: float

# Define the function to plot the cables power vs total power
def plot_cables_vs_total_power(data: List[StageData]):
    # Define the figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))

    # Set the width of the bars
    bar_width = 0.15
    opacity = 0.8

    # Get the unique temperatures and n_q values for plotting
    temperatures = sorted(set(d.temperature for d in data))
    n_q_values = sorted(set(d.n_q for d in data))
    
    # Plot each set of data
    for i, n_q in enumerate(n_q_values):
        cables_power = [d.cables_power for d in data if d.n_q == n_q]
        cryo_power = [d.cryo_power for d in data if d.n_q == n_q]
        
        # Calculate the positions for the bars
        offset = (bar_width + bar_width) * i
        bar_positions = np.arange(len(temperatures)) + offset
        
        # Plot the bars for cables power and total power
        ax.bar(bar_positions, cables_power, bar_width, align='center', alpha=opacity, label=f'Cables Power (n_q={n_q})')
        ax.bar(bar_positions + bar_width, cryo_power, bar_width, align='center', alpha=opacity, label=f'Cryo Power (n_q={n_q})')

    # Set labels, title, and legend
    ax.set_xlabel('Temperature of Qubit / K')
    ax.set_ylabel('Power Consumption / W')
    ax.set_title('Cables Power vs Total Power of Spin Qubit')
    ax.set_xticks(np.arange(len(temperatures)) + bar_width * len(n_q_values) / 2)
    ax.set_xticklabels([f'{temp:.3f}' for temp in temperatures])
    ax.set_yscale('log')
    ax.legend()

    # Save the plot as an SVG file
    plt.savefig('Cables_vs_Total_Power_3.svg', format='svg')

    # Show the plot
    plt.show()

# Example usage
if __name__ == "__main__":
    # Define example data
    temps = [0.007, 0.1, 0.8, 4, 50, 300]
    attens = [0.0, 0.0, 0.0, 3, 0.0, 0.0]
    Tq_values = [0.006, 0.02, 0.04, 0.1, 0.3, 1]
    n_q_values = [1, 50, 100, 500]
    silicon_abs = 0.0
    f = 39.33
    rabi = 0.5

    example_data = []
    for Tq in Tq_values:
        for n_q in n_q_values:
            spin_qubit = SpinQubit(
                n_q=n_q,
                Tq=Tq,
                f=f,
                rabi=rabi,
                rabi_in_MHz = rabi*1e6,
                f_in_GHz = f*1e-9,
                atts_list=attens,
                stages_ts=temps,
                silicon_abs=silicon_abs,
                efficiency='Carnot'
            )
            logger.info(f"Number of qubits: {spin_qubit.n_q}")
            logger.info(f"SpinQubit temp: {spin_qubit.Tq}")
            logger.info(f"SpinQubit cable_power: {spin_qubit.cables_power()}")
            logger.info(f"SpinQubit cryo_power: {spin_qubit.cryo_power()}")
            logger.info(f"SpinQubit total_power: {spin_qubit.total_power()}")
            cables_power = spin_qubit.cables_power()
            cryo_power = spin_qubit.cryo_power()
            example_data.append(StageData(temperature=Tq, n_q=n_q, cables_power=cables_power, cryo_power=cryo_power))

    # Plot the example data with uniform bar widths on a logarithmic scale
    plot_cables_vs_total_power(example_data)
