from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from numpy.polynomial.polynomial import Polynomial
# from pandas import DataFrame
from pydantic import BaseModel, Field, confloat, conint, root_validator
from scipy.constants import h, k

from spin_qe.components.cryostat import Cryo


class SpinQubit(BaseModel):
    n_q: int = conint(ge=1, le=20)
    Tq: float = confloat(gt=0.0, le=300)
    f: float = Field(39.33e9, alias='f_in_GHz')
    rabi: float = Field(0.5e6, alias='rabi_in_MHz')
    atts_list: List[float] = []
    stages_ts: List[float] = []
    silicon_abs: float = confloat(ge=0.0, le=1.0)
    gate_t: Optional[float] = None  # Initialize as None
    gamma: float = Field(default_factory=lambda: 1.1 * 1e-5)
    cryostat: Optional[Cryo] = None  # Initialize as None
    echo: Optional[bool] = True # Initialize as None

    @root_validator(pre=True, skip_on_failure=True)
    def calculate_gate_t_and_gamma(cls, values):
        rabi = values.get('rabi')
        if rabi is not None:
            values['gate_t'] = 1e-6 / (2 * rabi)
        return values

    @root_validator(pre=True)
    def initialize_cryostat(cls, values):
        atts_list = values.get('atts_list')
        stages_ts = values.get('stages_ts')
        Tq = values.get('Tq')
        silicon_abs = values.get('silicon_abs')

        # Assuming Cryostat class takes these parameters for initialization
        cryostat = Cryo(temps=stages_ts, attens=atts_list,
                        Tq=Tq, Si_abs=silicon_abs)
        values['cryostat'] = cryostat
        return values

    def pow_1q(self):
        if self.gate_t is None:
            raise ValueError("gate_t is not set.")
        power = (np.pi**2) * (h * self.f) / (4 * self.gamma * self.gate_t**2)
        return power

    def total_power(self) -> float:
        power_at_Tq = self.pow_1q()
        input_power = self.cryostat.calculate_input_power(power_at_Tq)
        total_power = self.cryostat.total_power(input_power)
        return total_power
    

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

    def fid_1q(self, T2='nan') -> float:
        if T2 == 'nan':
            if self.echo:
                T2 = self.T2HQ()
                print(f"T2 echo: {T2}")
            else:
                T2 = self.T2Q()
        # T2 = (self.T2Q()+self.T2HQ())/2
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

###interesting
    fidelities = []
    temperatures = np.linspace(0.006, 1, 10)
    for temp in temperatures:
        spin_qubit_params = {
            'n_q': 5,
            'Tq': temp,
            'f': 39.33,
            'rabi_in_MHz': 0.6e6,
            'rabi': 0.6,
            'atts_list': [3, 0],
            'stages_ts': [4, 300],
            'silicon_abs': 0.0
        }
        spin_qubit = SpinQubit(**spin_qubit_params)
        fidelities.append(spin_qubit.fid_1q())

    
    plt.figure(figsize=(10, 6))
    plt.plot(temperatures, fidelities, label='Fidelity vs. Temperature')
    plt.xlabel('Temperature (K)')
    plt.ylabel('Fidelity')
    plt.title('2-Qubit Fidelity as a Function of Temperature')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
