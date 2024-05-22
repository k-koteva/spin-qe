from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from numpy.polynomial.polynomial import Polynomial
# from pandas import DataFrame
from pydantic import BaseModel, Field, confloat, conint, root_validator
from scipy.constants import h, k

from spin_qe.components.cables import sum_conduction_power
from spin_qe.components.cryostat import Cryo


class SpinQubit(BaseModel):
    n_q: int = Field(default=1, ge=1, le=10000000)
    Tq: float = Field(..., gt=0.0, le=300)
    rabi_in_MHz: float = Field(..., gt=0.0)
    rabi: float = Field(..., gt=0.0)  # This will be set in the validator
    f_in_GHz: float = Field(..., gt=0.0)   # Input in GHz
    f: float = Field(..., gt=0.0) 
    atts_list: List[float] = []
    stages_ts: List[float] = []
    silicon_abs: float = Field(default=0.0, ge=0, le=1)
    gate_t: Optional[float] = None  # Initialize as None
    gamma: float = Field(default_factory=lambda: 1.1 * 1e-5)
    cryostat: Optional[Cryo] = None  # Initialize as None
    efficiency: Optional[str] = 'Carnot' # Initialize as Carnot
    echo: Optional[bool] = True # Initialize as None
    
    @root_validator(pre=True)
    def calculate_gate_t_and_rabi(cls, values):
        rabi_in_MHz = values.get('rabi_in_MHz')
        rabi = values.get('rabi')
        f_in_GHz = values.get('f_in_GHz')
        f = values.get('f')
        
        # Convert f_in_GHz to f in Hz if provided
        if f_in_GHz is not None:
            values['f'] = f_in_GHz * 1e9
        elif f is not None:
            values['f'] = f
        # Convert rabi_in_MHz to rabi in Hz if provided
        if rabi_in_MHz is not None:
            values['rabi'] = rabi_in_MHz * 1e6
        elif rabi is not None:
            values['rabi'] = rabi

        # Calculate gate_t based on rabi in Hz
        if values.get('rabi') is not None:
            logger.info(f"Rabi: {values['rabi']}")
            values['gate_t'] = 1 / (2 * values['rabi'])
        return values
    

    @root_validator(pre=True)
    def initialize_cryostat(cls, values):
        atts_list = values.get('atts_list')
        stages_ts = values.get('stages_ts')
        Tq = values.get('Tq')
        silicon_abs = values.get('silicon_abs')

        # Assuming Cryostat class takes these parameters for initialization
        cryostat = Cryo(temps=stages_ts, attens=atts_list,
                        Tq=Tq, Si_abs=silicon_abs, cables_atten=30, efficiency=values.get('efficiency'))
        values['cryostat'] = cryostat
        return values

    def pow_1q(self):
        if self.gate_t is None:
            raise ValueError("gate_t is not set.")
        power = (np.pi**2) * (h * self.f) / (4 * self.gamma * self.gate_t**2)
        return power

    def cryo_power(self) -> float:
        power_at_Tq = self.pow_1q()
        if self.cryostat is None:
            raise ValueError("cryostat is not set.")
        input_power = self.cryostat.calculate_input_power(power_at_Tq)
        total_power = self.cryostat.total_power(input_power)
        return total_power
    
    def n_cables(self) -> int:
        return 6*self.n_q -1
    
    def cables_power(self) -> float:
        if self.cryostat is None:
            raise ValueError("cryostat is not set.")
        return sum_conduction_power(self.cryostat)*self.n_cables()
    
    def total_power(self) -> float:
        return self.cryo_power() + self.cables_power()

    

    # def T2Q(self) -> float:
    #     return 1.7e-5/(1+9*self.Tq)
    
    def T2Q(self) -> float:
        coefs = [ 4.00491793, -4.18169044,  3.49715485, -1.0763837 ]
        poly = Polynomial(coefs)
        print('Hanh')
        return poly(self.Tq)*1e-6
    
    def T2HQ(self) -> float:
        coefs = [116.75220984, -135.6927179, 56.56474202, -5.13410108]
        poly = Polynomial(coefs)
        print('Hanh')
        return poly(self.Tq)*1e-6
    
    def noiseFid(self, T2='nan', numiter=100) -> float:
        if T2=='nan':
            T2=self.T2HQ()
            print(f"T2 noisy: {T2}")
        else:
            T2 = float(T2)
        if T2 <= 0:
            T2 = 1e-20  # Set a small positive value for T2 to avoid division by zero or negative scale
        noise = np.random.normal(loc=self.f, scale=(1/T2), size=numiter)
        print(f"Friquency: {self.f}")
        print(f"Drive: {noise}")
        fids = []
        for n in noise:
            print()
            print(f"fidelity: {self.FidQ(n)}")
            fids.append(self.FidQ(n))
        return np.mean(fids)
    
    def noiseFidarray(self, T2='nan', numiter=100) -> List:
        if T2=='nan':
            T2=self.T2HQ()
            print(f"T2 noisy: {T2}")
        else:
            T2 = float(T2)
        if T2 <= 0:
            T2 = 1e-20  # Set a small positive value for T2 to avoid division by zero or negative scale
        noise = np.random.normal(loc=self.f, scale=(1/T2), size=numiter)
        print(f"Friquency: {self.f}")
        print(f"Drive: {noise}")
        fids = []
        for n in noise:
            print()
            print(f"fidelity: {self.FidQ(n)}")
            fids.append(self.FidQ(n)*100)
        return fids
    
    def FidQ(self, drive):
    # [rabi] = kHz, [B] = GH, [pert] = GHz, [time] = microsec
        time = np.pi/(2*self.rabi)
        rabi = self.rabi
        print(f"given f: {self.f}")
        delta = self.f - drive
        print(f"delta: {delta}")
        sq = (rabi)**2 + (delta)**2
        print(f"sq: {sq}")
        prob = ((rabi**2)*np.sin(time*np.sqrt(sq))**2)/sq
        return prob

    def fid_1q(self, T2=0) -> float:
        if T2 == 0:
            if self.echo:
                T2 = self.T2HQ()
                print(f"T2 echo: {T2}")
            else:
                T2 = self.T2Q()
        # T2 = (self.T2Q()+self.T2HQ())/2
        if T2 <= 0:
            raise ValueError("T2 must be positive")
        rabi = self.rabi
        print(f"rabi: {rabi}")
        logger.info(f"rabi: {rabi}")
        delta = 1/T2
        prob = 1 - (np.pi+1)*delta**2/(4*rabi**2)
        return prob

    def fid_2q_old(self) -> float:
        Fidelity = 1 - \
            (0.22 * (0.6 * ((0.1 + 0.9 * self.Tq) ** 2) + 0.4 * self.Tq))
        return Fidelity

    def fid_meas(self) -> float:
        # to import from qs_energetics
        return 0.999
    
    def fid_2q(self) -> float:
        a = -2.135
        b = 0.016
        c = 99.820

        # Calculate the fidelity using the quadratic function
        Fidelity = a * self.Tq**2 + b * self.Tq + c
        return Fidelity

    def fid_circ(self, num_1q_gates: int, num_2q_gates: int, num_meas: int = 0) -> float:
        """
        Calculate the total fidelity based on the number of 1q gates, 2q gates, and measurements.

        :param num_1q_gates: Number of one-qubit gates.
        :param num_2q_gates: Number of two-qubit gates.
        :param num_measurements: Number of measurements.
        :return: Total fidelity.
        """
        fid_1q = self.fid_1q()  # Assuming you have a method for 1q gate fidelity
        fid_2q = self.fid_2q()
        fid_meas = self.fid_meas()

        total_fid = (fid_1q ** num_1q_gates) * (fid_2q **
                                                num_2q_gates) * (fid_meas ** num_meas)
        # total_fid = (fid_1q ** num_1q_gates) 
        return total_fid

    class Config:
        arbitrary_types_allowed = True

def plot_fid_2q_temperature_range(qubit: SpinQubit, temp_range: List[float]):
    fidelities = []
    temperatures = np.linspace(temp_range[0], temp_range[1], 100)
    for temp in temperatures:
        qubit.Tq = temp
        fidelities.append(qubit.fid_2q())
    
    plt.figure(figsize=(10, 6))
    plt.plot(temperatures, fidelities, label='Fidelity vs. Temperature')
    plt.xlabel('Temperature (K)')
    plt.ylabel('Fidelity')
    plt.title('2-Qubit Fidelity as a Function of Temperature')
    plt.legend()
    plt.grid(True)
    plt.show()

class StageData(BaseModel):
    temperature: float
    n_q: int
    cables_power: float
    cryo_power: float
    total_power: float


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
    ax.set_title('Cables Power vs Cryo Power of Spin Qubit')
    ax.set_xticks(np.arange(len(temperatures)) + bar_width * len(n_q_values) / 2)
    ax.set_xticklabels([f'{temp:.3f}' for temp in temperatures])
    ax.set_yscale('log')
    ax.legend()

    # Save the plot as an SVG file
    plt.savefig('Cables_vs_Total_Power_3Carnot.svg', format='svg')

    # Show the plot
    plt.show()

def plot_cables_vs_total_power_ratio(data: List[StageData]):
    # Define the figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))

    # Set the width of the bars
    bar_width = 0.15
    opacity = 0.8

    # Get the unique temperatures and n_q values for plotting
    temperatures = sorted(set(d.temperature for d in data))
    n_q_values = sorted(set(d.n_q for d in data))
    
    # Plot each set of data as a ratio of cryo power to cables power
    for i, n_q in enumerate(n_q_values):
        cables_power = [d.cables_power for d in data if d.n_q == n_q]
        cryo_power = [d.cryo_power for d in data if d.n_q == n_q]
        power_ratio = [cryo / cables for cryo, cables in zip(cryo_power, cables_power) if cables != 0]
        
        # Calculate the positions for the bars
        offset = (bar_width + bar_width) * i
        bar_positions = np.arange(len(power_ratio)) + offset
        
        # Plot the bars for the ratio of cryo power to cables power
        ax.bar(bar_positions, power_ratio, bar_width, align='center', alpha=opacity, label=f'Power Ratio (n_q={n_q})')

    # Set labels, title, and legend
    ax.set_xlabel('Temperature of Qubit / K')
    ax.set_ylabel('Power Ratio (Cryo Power / Cables Power)')
    ax.set_title('Power Ratio of Cryo Power to Cables Power of Spin Qubit')
    ax.set_xticks(np.arange(len(temperatures)) + bar_width * len(n_q_values) / 2)
    ax.set_xticklabels([f'{temp:.3f}' for temp in temperatures])
    ax.set_yscale('log')
    ax.legend()

    # Save the plot as an SVG file
    plt.savefig('Cables_vs_Total_Power_RatioSmallDev.svg', format='svg')

    # Show the plot
    plt.show()

# Example usage:
# Assuming you have an instance of SpinQubit initialized, you can call:
# plot_fid_2q_temperature_range(spin_qubit, [0, 300])


def main():

    # spin_qubit_params = {
    #     'n_q': 5,
    #     'Tq': 0.04,
    #     'f': 39.33,
    #     'rabi_in_MHz': 0.6e6,
    #     'rabi': 0.6,
    #     'atts_list': [3, 0],
    #     'stages_ts': [4, 300],
    #     'silicon_abs': 0.01
    # }

    # # Create a SpinQubit instance
    # spin_qubit = SpinQubit(**spin_qubit_params)
    # logger.info(f"SpinQubit rabi: {spin_qubit.rabi}")
    # #Accessing properties of the SpinQubit instance
    # print(f"Gate Time: {spin_qubit.gate_t} seconds")
    # print(f"Gamma: {spin_qubit.gamma}")
    # print(f"Cryostat Stages DataFrame:\n{spin_qubit.cryostat.stages}")
    # # Print the results
    # print(f"Power at Tq: {spin_qubit.pow_1q()}")
    # print(f"Total Power: {spin_qubit.total_power()}")
    # print(f"Single-qubit gate fidelity: {spin_qubit.fid_1q()}")
    # print(f"Two-qubit gate fidelity: {spin_qubit.fid_2q()}")
    # print(f"Measurement fidelity: {spin_qubit.fid_meas()}")

    # total_fid = spin_qubit.fid_circ(
    #     num_1q_gates=10, num_2q_gates=5, num_meas=3)
    # print(f"Total fidelity: {total_fid}")

    
### CARNOT EFFICIENCY
    # fidelities = []
    # example_data = []
    # temperatures = [0.007, 0.1, 0.8, 4, 50, 300]
    # for number_q in [1,10,20,50]:
    #     for temp in temperatures:
    #         spin_qubit_params = {
    #             'n_q': number_q,
    #             'Tq': temp,
    #             'f': 39.33,
    #             'rabi_in_MHz': 1.2e6,
    #             'rabi': 1.2,
    #             'atts_list': [3, 0],
    #             'stages_ts': [4, 300],
    #             'silicon_abs': 0.0,
    #             'efficiency':'Carnot'
    #         }
    #         spin_qubit = SpinQubit(**spin_qubit_params)
    #         logger.info(f"Number of qubits: {number_q}")
    #         logger.info(f"SpinQubit temp: {spin_qubit_params['Tq']}")
    #         logger.info(f"SpinQubit cable_power: {spin_qubit.cables_power()}")
    #         logger.info(f"SpinQubit cryo_power: {spin_qubit.cryo_power()}")
    #         logger.info(f"SpinQubit total_power: {spin_qubit.total_power()}")
    #         example_data.append(StageData(temperature=spin_qubit.Tq, n_q=spin_qubit.n_q, cables_power=spin_qubit.cables_power(), cryo_power=spin_qubit.cryo_power(), total_power=spin_qubit.total_power()))

    # # Plot the example data with uniform bar widths on a logarithmic scale
    # plot_cables_vs_total_power(example_data)

    ### SMALL SYSTEM EFFICIENCY
    # example_data = []
    # temperatures = [0.007, 0.1, 0.8, 4, 50, 300]
    # for number_q in [1,20,30]:
    #     for temp in temperatures:
    #         spin_qubit_params = {
    #             'n_q': number_q,
    #             'Tq': temp,
    #             'f': 39.33,
    #             'rabi_in_MHz': 0.6e6,
    #             'rabi': 0.6,
    #             'atts_list': [3, 0],
    #             'stages_ts': [4, 300],
    #             'silicon_abs': 0.0,
    #             'efficiency':'Small System'
    #         }
    #         spin_qubit = SpinQubit(**spin_qubit_params)
    #         logger.info(f"Number of qubits: {number_q}")
    #         logger.info(f"SpinQubit temp: {spin_qubit_params['Tq']}")
    #         logger.info(f"SpinQubit cable_power: {spin_qubit.cables_power()}")
    #         logger.info(f"SpinQubit cryo_power: {spin_qubit.cryo_power()}")
    #         logger.info(f"SpinQubit total_power: {spin_qubit.total_power()}")
    #         example_data.append(StageData(temperature=spin_qubit.Tq, n_q=spin_qubit.n_q, cables_power=spin_qubit.cables_power(), cryo_power=spin_qubit.cryo_power(), total_power=spin_qubit.total_power()))

    # # Plot the example data with uniform bar widths on a logarithmic scale
    # # plot_cables_vs_total_power(example_data)

    # plot_cables_vs_total_power_ratio(example_data)



    # plt.figure(figsize=(10, 6))
    # plt.plot(temperatures, fidelities, label='Fidelity vs. Temperature')
    # plt.xlabel('Temperature (K)')
    # plt.ylabel('Fidelity')
    # plt.title('2-Qubit Fidelity as a Function of Temperature')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    spin_qubit_params = {
                'n_q': 1,
                'Tq': 0.04,
                'f': 39.33,
                'rabi_in_MHz': 0.7,
                # 'rabi': 0.7e6,
                'atts_list': [3, 0],
                'stages_ts': [4, 300],
                'silicon_abs': 0.0,
                'efficiency':'Small System'
            }
    spin_qubit = SpinQubit(**spin_qubit_params)
    logger.info(f"Number of qubits: {spin_qubit.n_q}")
    logger.info(f"SpinQubit temp: {spin_qubit_params['Tq']}")
    logger.info(f"SpinQubit cable_power: {spin_qubit.cables_power()}")
    logger.info(f"SpinQubit cryo_power: {spin_qubit.cryo_power()}")
    logger.info(f"SpinQubit total_power: {spin_qubit.total_power()}")
    logger.info(f"SpinQubit Rabi: {spin_qubit.rabi}")
    logger.info(f"SpinQubit Gate Time: {spin_qubit.gate_t}")


if __name__ == "__main__":
    main()
