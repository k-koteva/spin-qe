from typing import List

import numpy as np
from matplotlib import pyplot as plt
from pydantic import BaseModel, validator

from spin_qe.components.cryostat import Cryo


# Define a Pydantic model for stage data
class StageData(BaseModel):
    temperature: float
    heat_extracted: List[float]
    power_consumption: List[float]

    # Custom validator to ensure lists have the same length
    @validator('power_consumption')
    def lists_must_be_same_length(cls, v, values):
        if 'heat_extracted' in values and len(v) != len(values['heat_extracted']):
            raise ValueError('The length of heat_extracted and power_consumption must be the same')
        return v

temps = [0.007, 0.1, 0.8, 4, 50, 300]
attens = [20, 20.0, 10, 20, 0.0, 0.0]
input_power = 1.0
Tq_values = [0.006, 0.02, 0.04, 0.1, 0.3, 1]
reference_powersCarnot = []
reference_powers = []
example_data = []
for Tq in Tq_values:
    cryo = Cryo(Tq=Tq, temps=temps, attens=attens, Si_abs=0.0, cables_atten=30, efficiency= 'Carnot')
    reference_powerCarnot = cryo.total_power(power=input_power)
    cryo = Cryo(Tq=Tq, temps=temps, attens=attens, Si_abs=0.0, cables_atten=30, efficiency='Small System')
    reference_power = cryo.total_power(power=input_power)
    example_data.append(StageData(temperature=Tq, heat_extracted=[reference_power], power_consumption=[reference_powerCarnot]))
# Example data for plotting (this should be replaced with the actual data)
# example_data = [
#     StageData(temperature=0.006, heat_extracted=[1e2], power_consumption=[1e1]),
#     StageData(temperature=0.02, heat_extracted=[1e4], power_consumption=[1e3]),
#     StageData(temperature=0.1, heat_extracted=[1e6], power_consumption=[1e5]),
#     StageData(temperature=1, heat_extracted=[1e8], power_consumption=[1e7])
# ]

def plot_log_data_uniform_bars(data: List[StageData]):
    # Define the figure and axis
    fig, ax = plt.subplots()

    # Set the width of the bars
    bar_width = 0.15
    opacity = 0.8

    # Calculate the positions for each set of bars
    # Use a log scale for the x positions to ensure bars have the same visual width on the log scale
    temperatures = np.array([d.temperature for d in data])
    log_temperatures = np.log10(temperatures)
    positions = [10**(log_temp - bar_width/2) for log_temp in log_temperatures]
    
    # Plot each set of data
    for i in range(len(data[0].heat_extracted)):
        heat_bars = [d.heat_extracted[i] for d in data]
        power_bars = [d.power_consumption[i] for d in data]
        
        # Offset the positions for each set of bars
        offset = (bar_width + bar_width) * i
        bar_positions = [10**(np.log10(pos) + offset) for pos in positions]
        
        # Plot the bars for heat extracted and power consumption
        ax.bar(bar_positions, heat_bars, bar_width * temperatures, align='edge', alpha=opacity, color='b', label='Small System efficiency')
        ax.bar(bar_positions, power_bars, bar_width * temperatures, align='edge', alpha=opacity, color='r', label='Carnot efficiency')

    # ax.legend([f"Temp: {d.temperature}K, Atten: {d.attenuation}dB, Cables Atten: {d.cables_atten}dB" for d in data], loc='upper right')

    # Set labels, title, and legend
    ax.set_xlabel('Temperature of qubit / K')
    ax.set_ylabel('Power Consumption / W')
    ax.set_title('Power Consumption of Cryostat')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()

    plt.savefig('Cryotat_Consumption_20_20_10_20_db.svg', format='svg')
    # Show the plot
    plt.show()

# Plot the example data with uniform bar widths on a logarithmic scale
plot_log_data_uniform_bars(example_data)



def main():
    # Example usage
    # plot_data(example_data)
    print("Hello, World!")
    print(reference_powers)

if __name__ == "__main__":
    main()
