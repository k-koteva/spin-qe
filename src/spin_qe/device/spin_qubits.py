from typing import List, Optional

import numpy as np
from loguru import logger
# from pandas import DataFrame
from pydantic import BaseModel, Field, confloat, conint, root_validator
from scipy.constants import h

from spin_qe.components.cables import sum_conduction_power, sum_emptycryo_power
from spin_qe.components.cryostat import Cryo
from numpy.polynomial.polynomial import Polynomial
wire_type = "304SS_UT_S"
class SpinQubit(BaseModel):
    n_q: int = conint(ge=1, le=10000000000000000)
    depth: int = conint(ge=1, le=10000000000000000)
    Tq: float = confloat(gt=0.0, le=300)
    f: float = Field(39.33e9, alias='f_in_GHz')
    rabi: float = Field(0.5e6, alias='rabi_in_MHz')
    atts_list: List[float] = []
    stages_ts: List[float] = []
    silicon_abs: float = confloat(ge=0.0, le=1.0)
    gate_t: Optional[float] = None  # Initialize as None
    gamma: float = Field(default_factory=lambda: 1.1 * 1e-5)
    cryostat: Optional[Cryo] = None  # Initialize as None
    efficiency: Optional[str] = 'Small system'

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
                        Tq=Tq, Si_abs=silicon_abs, cables_atten=30, efficiency=values.get('efficiency'))
        values['cryostat'] = cryostat
        return values

    def pow_1q(self):
        if self.gate_t is None:
            raise ValueError("gate_t is not set.")
        power = (np.pi**2) * (h * self.f) / (4 * self.gamma * self.gate_t**2)
        return power

    def input_power(self) -> float:
        power_at_Tq = self.pow_1q()
        if self.cryostat is None:
            raise ValueError("cryostat is not set.")
        input_power = self.cryostat.calculate_input_power(power_at_Tq)
        return input_power
        # return input_power*1000

    def cryo_power(self) -> float:
        power_at_Tq = self.pow_1q()
        if self.cryostat is None:
            raise ValueError("cryostat is not set.")
        input_power = self.cryostat.calculate_input_power(power_at_Tq)
        total_power = self.cryostat.total_power(input_power)
        return total_power
        # return input_power*1000
    
    def n_cables(self) -> int:
        logger.info(f"n_q: {self.n_q}")
        logger.info(f"number of cables: {6*self.n_q -1}")
        # return self.n_q
        return 5*self.n_q -1
    
    def cables_power(self) -> float:
        if self.cryostat is None:
            raise ValueError("cryostat is not set.")
        return sum_conduction_power(self.cryostat, wire_type=wire_type)*self.n_cables()
    
    def single_cables_power(self) -> float:
        if self.cryostat is None:
            raise ValueError("cryostat is not set.")
        return sum_conduction_power(self.cryostat, wire_type=wire_type)*self.n_cables()
    
    def empty_cryo_power_carnot(self) -> float:
        if self.cryostat is None:
            raise ValueError("cryostat is not set.")
        return sum_emptycryo_power(self.cryostat)
    
    def empty_cryo_energy_real(self) -> float:
        if self.Tq <= 0.001:
            e_empty = (250*(0.000000000001)**(-3/4) + 2000)
        else:
            e_empty = (250*(self.Tq-0.001)**(-3/4) + 2000)
        return e_empty
    
    def empty_cryo_power(self) -> float:
        if self.cryostat is None:
            raise ValueError("cryostat is not set.")
        if self.efficiency == 'Carnot':
            return self.empty_cryo_power_carnot()
        else:
            return self.empty_cryo_energy_real()
    
    def generator_power(self) -> float:
        power_at_Tq = self.pow_1q()
        if self.cryostat is None:
            raise ValueError("cryostat is not set.")
        input_power = self.cryostat.calculate_input_power(power_at_Tq)
        total_power = self.cryostat.total_power(input_power)
        return input_power*1000
    
    def total_power(self) -> float:
        return self.cryo_power() + self.cables_power() + self.generator_power() + self.empty_cryo_power()

    def T2Q(self) -> float:
        return 1.7e-5/(1+9*self.Tq)

    def fid_1q(self, T2='nan') -> float:
        if T2 == 'nan':
            T2 = self.T2Q()
        rabi = self.rabi
        logger.info(f"rabi: {rabi}")
        delta = 1/T2
        prob = 1 - (np.pi+1)*delta**2/(4*rabi**2)   
        return prob

    def fid_2q(self) -> float:
        Fidelity = 1 - \
            (0.22 * (0.6 * ((0.1 + 0.9 * self.Tq) ** 2) + 0.4 * self.Tq))
        return Fidelity

    def fid_meas(self) -> float:
        # to import from qs_energetics
        return 0.999

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
        return total_fid
    
    def fid_circ_nomeas(self, num_1q_gates: int, num_2q_gates: int, num_meas: int = 0) -> float:
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
                                                num_2q_gates)
        return total_fid
    
    #### UP TO DATE
    def fidelity(self, model='Slow')->float:
        fit_coeff_drive = np.array([11.4, 1.77, 0, 0, 6.01])
        fit_coeff_measurement = 10**(-3)*np.array([3, 0, 0, 0, 0, 0, 0, 0, 3.3])
        drive_noise_fit = Polynomial(fit_coeff_drive)
        meas__noise_fit = Polynomial(fit_coeff_measurement)
        if model == 'Slow':
            # Rabi**4
            # fidelity = 1-self.n_q*((self.depth**2)*1.44*(10**20)*(drive_noise_fit(self.Tq)/self.rabi**4))
            # logger.info(f"fid: {meas__noise_fit(self.Tq)}")
            fidelity = 1-self.n_q*((self.depth**2)*1.44*(10**20)*(drive_noise_fit(self.Tq)/self.rabi**4) + meas__noise_fit(self.Tq))
            # fidelity = 1-self.n_q*meas__noise_fit(self.Tq) - self.n_q*((self.depth**2)*144*drive_noise_fit(self.Tq)/self.rabi)
        elif model == 'Markov':
            # Rabi
            # fidelity = 1-self.n_q*(self.depth*144*drive_noise_fit(self.Tq)/self.rabi)
            # logger.info(f"fid: {meas__noise_fit(self.Tq)}")
            fidelity = 1-self.n_q*(self.depth*144*drive_noise_fit(self.Tq)/self.rabi + meas__noise_fit(self.Tq))

            # fidelity = 1-self.n_q*meas__noise_fit(self.Tq) - self.n_q*(self.depth*144*drive_noise_fit(self.Tq)/self.rabi)
        else:
            raise ValueError("Invalid trend value. Please choose either 'slow' or 'markov'.")

        return fidelity
    
    def fidelity_nomeas(self, model='Slow')->float:
        fit_coeff_drive = np.array([11.4, 1.77, 0, 0, 6.01])
        # fit_coeff_measurement = 10**(-3)*np.array([3, 0, 0, 0, 0, 0, 0, 0, 3.3])
        drive_noise_fit = Polynomial(fit_coeff_drive)
        # meas__noise_fit = Polynomial(fit_coeff_measurement)
        if model == 'Slow':
            # Rabi**4
            # fidelity = 1-self.n_q*((self.depth**2)*1.44*(10**20)*(drive_noise_fit(self.Tq)/self.rabi**4))
            # logger.info(f"fid: {meas__noise_fit(self.Tq)}")
            fidelity = 1-self.n_q*((self.depth**2)*1.44*(10**20)*(drive_noise_fit(self.Tq)/self.rabi**4))
            # fidelity = 1 - self.n_q*((self.depth**2)*144*drive_noise_fit(self.Tq)/self.rabi)
        elif model == 'Markov':
            # Rabi
            # fidelity = 1-self.n_q*(self.depth*144*drive_noise_fit(self.Tq)/self.rabi)
            # logger.info(f"fid: {meas__noise_fit(self.Tq)}")
            fidelity = 1-self.n_q*(self.depth*144*drive_noise_fit(self.Tq)/self.rabi)

            # fidelity = 1 - self.n_q*(self.depth*144*drive_noise_fit(self.Tq)/self.rabi)
        else:
            raise ValueError("Invalid trend value. Please choose either 'slow' or 'markov'.")

        return fidelity

    class Config:
        arbitrary_types_allowed = True


def main():

    spin_qubit_params = {
        'n_q': 5,
        'Tq': 0.04,
        'f': 39.33,
        'rabi_in_MHz': 0.6e6,
        'rabi': 0.6,
        'atts_list': [3, 0],
        'stages_ts': [4, 300],
        'silicon_abs': 0.01,
        'efficiency': 'Carnot'
    }

    # Create a SpinQubit instance
    spin_qubit = SpinQubit(**spin_qubit_params)
    logger.info(f"SpinQubit rabi: {spin_qubit.rabi}")
    #Accessing properties of the SpinQubit instance
    print(f"Gate Time: {spin_qubit.gate_t} seconds")
    print(f"Gamma: {spin_qubit.gamma}")
    print(f"Cryostat Stages DataFrame:\n{spin_qubit.cryostat.stages}")
    # Print the results
    print(f"Power at Tq: {spin_qubit.pow_1q()}")
    print(f"Total Power: {spin_qubit.total_power()}")
    print(f"Single-qubit gate fidelity: {spin_qubit.fid_1q()}")
    print(f"Two-qubit gate fidelity: {spin_qubit.fid_2q()}")
    print(f"Measurement fidelity: {spin_qubit.fid_meas()}")

    total_fid = spin_qubit.fid_circ(
        num_1q_gates=10, num_2q_gates=5, num_meas=3)
    print(f"Total fidelity: {total_fid}")
    logger.info(f"Number of qubits: {spin_qubit.n_q}")
    logger.info(f"Number of cables: {spin_qubit.n_cables()}")
    logger.info(f"Power in cables: {spin_qubit.cables_power()}")


if __name__ == "__main__":
    main()
