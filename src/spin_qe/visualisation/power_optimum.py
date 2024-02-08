
from typing import Any, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from numpy.polynomial.polynomial import Polynomial
from pydantic import BaseModel

from spin_qe.device.spin_qubitsGlobal import SpinQubit


def rabi_ofMT(Time,fid=0.99):
    my_rabi = np.sqrt((np.pi+1)/(1-fid))/(2*Time)
    return my_rabi

def mynew_func(Temp, fidelity=0.99):
    spin_qubit_params = {
        'n_q': 1,
        'Tq': Temp,
        'f': 11.20,
        'rabi_in_MHz': 0.3e6,
        'rabi': 0.3,
        'atts_list': [3, 0],
        'stages_ts': [4, 300],
        'silicon_abs': 0.0
    }
    spin_qubit = SpinQubit(**spin_qubit_params)
    myTime = spin_qubit.T2HQ()
    rabi_fixed = rabi_ofMT(myTime,fid=fidelity)
    spin_qubit_params = {
        'n_q': 1,
        'Tq': Temp,
        'f': 11.20,
        'rabi_in_MHz': rabi_fixed,
        'rabi': rabi_fixed*1e-6,
        'atts_list': [3, 0],
        'stages_ts': [4, 300],
        'silicon_abs': 0.0
    }
    spin_qubit = SpinQubit(**spin_qubit_params)
    return spin_qubit.total_power()



def find_min_and_index(values: list) -> Tuple[Any, int]:
    min_value = min(values)
    min_index = values.index(min_value)
    return min_value, min_index




def main():
    my_temp = np.linspace(0.03, 1, 150)
    powers_99 = [mynew_func(temp, fidelity=0.99) for temp in my_temp]
    powers_95 = [mynew_func(temp, fidelity=0.95) for temp in my_temp]
    powers_80 = [mynew_func(temp, fidelity=0.80) for temp in my_temp]

    plt.plot(my_temp, powers_99, label='power, fid=0.99')
    plt.plot(my_temp, powers_95, label='power, fid=0.95')
    plt.plot(my_temp, powers_80, label='power, fid=0.80')

    plt.xlabel('Temp')
    plt.ylabel('Cryo power')
    plt.legend()
    plt.savefig('power_vs_temp.pdf', format='pdf')
    plt.show()

    # Example usage
    values = [10, 20, 5, 30, 5]
    min_value, min_index = find_min_and_index(powers_95)
    print(f"Minimum value: {min_value}, located at index: {min_index}")
if __name__ == "__main__":
    main()
