import os
from typing import Any, List, Tuple, Union
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from matplotlib.colors import LogNorm

import spin_qe.device.spin_qubits as sq


def calculate_power_noise(efficiency: str, tqb: np.ndarray, rabifreq: np.ndarray, calculate_energy: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    attens = [0, 0, 3, 0, 0]
    stage_ts = [0.1, 0.8, 4, 50, 300]
    silicon_abs = 0.0

    powerGrid = np.zeros((len(rabifreq), len(tqb)))
    cryoGrid = np.zeros((len(rabifreq), len(tqb)))
    conductionGrid = np.zeros((len(rabifreq), len(tqb)))
    fidelityGrid = np.zeros((len(rabifreq), len(tqb)))
    energyGrid = np.zeros((len(rabifreq), len(tqb)))
    cryoEGrid = np.zeros((len(rabifreq), len(tqb)))
    conductionEGrid = np.zeros((len(rabifreq), len(tqb)))

    for i, rabi in enumerate(rabifreq):
        logger.info(f'Rabi frequency = {rabi}')
        for j, tq in enumerate(tqb):
            logger.info(f'Temperature = {tq}')
            mySQ = sq.SpinQubit(n_q=1, Tq=tq, rabi=rabi, rabi_in_MHz=rabi * 1e6, atts_list=attens, efficiency=efficiency,
                                stages_ts=stage_ts, silicon_abs=silicon_abs)
            cryoGrid[i, j] = mySQ.cryo_power()
            # conductionGrid[i, j] = mySQ.single_cables_power()
            # conductionGrid[i, j] = mySQ.empty_cryo_power()
            conductionGrid[i, j] = mySQ.empty_cryo_power()
            powerGrid[i, j] = cryoGrid[i, j] + conductionGrid[i, j]

            energyGrid[i, j] = powerGrid[i, j]*mySQ.gate_t
            cryoEGrid[i, j] = cryoGrid[i, j]*mySQ.gate_t
            conductionEGrid[i, j] = conductionGrid[i, j]*mySQ.gate_t
            # conductionEGrid[i, j] = conductionGrid[i, j]
            

            fidelityGrid[i, j] = mySQ.fid_1q()
            logger.info(f'fid = {mySQ.fid_1q()}')

    return powerGrid, fidelityGrid, conductionGrid, cryoGrid, energyGrid, conductionEGrid, cryoEGrid 

def optimize_power_and_efficiency(powerful: np.ndarray, smoothFid: np.ndarray, metric_target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    opt_power = 1e28 * np.ones(len(metric_target))
    gate_efficiency = np.zeros(len(metric_target))

    for k in range(len(metric_target)):
        for i in range(len(powerful)):
            for j in range(len(powerful[i])):
                if smoothFid[i, j] >= metric_target[k]:
                    if opt_power[k] > powerful[i, j]:
                        opt_power[k] = powerful[i, j]
        if opt_power[k] == 1e28:
            opt_power[k] = np.nan
        else:
            gate_efficiency[k] = metric_target[k] / opt_power[k]

    return opt_power, gate_efficiency

def plot_results(R: np.ndarray, T: np.ndarray, powerful: np.ndarray, smoothFid: np.ndarray, fid_levels: List[float], power_levels: List[float], filename: str, plot_energy: bool = False):
        # Define the path to the Results folder
    results_folder = os.path.expanduser('~/Dropbox/Apps/Overleaf/Energetics NISQ Spin qubits/Results/Plots1CPQ')
    
    # Create the Results folder if it doesn't exist
    os.makedirs(results_folder, exist_ok=True)
    
    # Join the path with the filename
    full_path = os.path.join(results_folder, filename)

    mkl_fid = {level: f'${level}$' for level in fid_levels}
    mkl_power = {level: f'${level}$' for level in power_levels}
    label = 'Energy /J' if plot_energy else 'Power /W'
    fig = plt.figure()
    ax0 = fig.gca()
    powerPlot = ax0.pcolor(R, T, powerful, 
                           norm=LogNorm(vmin=float(np.nanmin(powerful)), vmax=float(np.nanmax(powerful))),
                           shading="auto")

    smoothFid = np.where(smoothFid <= 0, 1e-10, smoothFid)
    
    fid_contour = ax0.contour(R, T, smoothFid, fid_levels,
                              norm=LogNorm(vmin=float(np.nanmin(smoothFid)) + 1e-10,
                                           vmax=float(np.nanmax(smoothFid)) + 1e-10),
                              colors="black")

    power_contour = ax0.contour(R, T, powerful, power_levels,
                                norm=LogNorm(vmin=float(np.nanmin(powerful)),
                                             vmax=float(np.nanmax(powerful))),
                                colors="white")

    ax0.clabel(fid_contour, levels=fid_levels, fmt=mkl_fid, fontsize=10, inline=True)
    ax0.clabel(power_contour, levels=power_levels, fmt=mkl_power, fontsize=10, inline=True)

    ax0.set_yscale('log')
    cbar = fig.colorbar(powerPlot, ax=ax0)
    ax0.set_ylabel('Temperature of quantum chip /K', fontsize=14)
    ax0.set_xlabel('Rabi frequency /MHz', fontsize=14)
    
    cbar.ax.set_ylabel(label, fontsize=12)
    fig.savefig(filename, bbox_inches='tight')

# def plot_power_vs_temperature(tqb: np.ndarray, powerCarnot: np.ndarray, powerSmall: np.ndarray, filename: str):
#     plt.figure()
#     plt.plot(tqb, powerCarnot, label='Small systems Efficiency')
#     # plt.plot(tqb, powerSmall, label='Small System Efficiency', linestyle='--')
#     plt.xlabel('Temperature of quantum chip /K')
#     plt.ylabel('Power /W')
#     plt.legend()
#     plt.savefig(filename, bbox_inches='tight')

# def plot_energy_vs_temperature(tqb: np.ndarray, energyCarnot: np.ndarray, filename: str):
#     plt.figure()
#     plt.plot(tqb, energyCarnot, label='Small Systems Efficiency')
#     plt.xlabel('Temperature of quantum chip /K')
#     plt.xscale('log')
#     plt.ylabel('Energy /J')
#     plt.legend()
#     plt.savefig(filename, bbox_inches='tight')

def plot_energy_vs_temperature(tqb: np.ndarray, energyCarnot: np.ndarray, energySmall: np.ndarray, filename: str):
    plt.figure()
    plt.plot(tqb, energyCarnot, linewidth=2.5)  # Thicker line, removed legend
    plt.plot(tqb, energySmall, linestyle='--', linewidth=2.5)  # Thicker line, removed legend
    plt.xlabel(r'Qubit temperature $T_q$ (K)', fontsize=16)  # Larger x-axis label
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel(r'$E_{cond}$ (Joules)', fontsize=16)  # Larger y-axis label
    plt.tick_params(axis='both', which='major', labelsize=16)  # Larger ticks
    plt.savefig(filename, bbox_inches='tight')

def find_min_and_index(values: list) -> Tuple[Any, int]:
    min_value = min(values)
    min_index = values.index(min_value)
    return min_value, min_index

def save_energy_data_to_csv(tqb: np.ndarray, energy_carnot: np.ndarray, energy_ss: np.ndarray, filename: str) -> None:
    # Create a DataFrame from the provided data
    data = {
        'Tqb': tqb,
        'energyCarnot': energy_carnot,
        'energySS': energy_ss
    }
    df = pd.DataFrame(data)

    # Save the DataFrame to a CSV file
    df.to_csv(filename, index=False)

def main():
    tqb = np.logspace(np.log10(0.0011), np.log10(10), 100)
    rabifreq = np.linspace(0.01, 15, 100)
    R, T = np.meshgrid(rabifreq, tqb, indexing='ij')
    eff = 'Carnot'
    powerful, smoothFid, conductionP, cryoP, energyTotal, conductionE, cryoE = calculate_power_noise(eff, tqb, rabifreq)
    
    metric_target = np.linspace(0.90, 0.999, 30)
    opt_power, gate_efficiency = optimize_power_and_efficiency(powerful, smoothFid, metric_target)
    
    fid_levels: List[float] = [0.90, 0.99, 0.997]
    power_levels: List[float] = [1, 10, 30]
    energy_levels: List[float] = [1e-8, 1e-7, 1e-6]
    # energy_levels: List[float] = [1e-6, 1e-5, 1e-4]
    # power_levels: List[float] = [1e2, 1e5, 1e7]
    # energy_levels: List[float] = [1e-3, 1e-1]
    # plot_results(R, T, powerful, smoothFid, fid_levels, power_levels, 'carnotspinQubit_optimPowSiAbs001realTRIAL_checked.pdf')
    # plot_results(R, T, conductionP, smoothFid, fid_levels, power_levels, 'carnotspinQubit_optimPowSiAbs001realTRIAL_checked_conduction.pdf')
    # plot_results(R, T, cryoP, smoothFid, fid_levels, power_levels, 'carnotspinQubit_optimPowSiAbs001realTRIAL_checked_cryo.pdf')
    # plot_results(R, T, energyTotal, smoothFid, fid_levels, energy_levels, 'carnotspin10Qubit_optimPowSiAbs001realTRIAL_checked_energy.pdf', plot_energy=True)
    # plot_results(R, T, conductionE, smoothFid, fid_levels, energy_levels, 'carnotspin10Qubit_optimPowSiAbs001realTRIAL_checked_conduction_energy.pdf', plot_energy=True)
    # plot_results(R, T, cryoE, smoothFid, fid_levels, energy_levels, 'carnotspin10Qubit_optimPowSiAbs001realTRIAL_checked_cryo_energy.pdf', plot_energy=True)

    #  Extract power values at Rabi frequency of 1 MHz

    rabi_index = np.argmin(np.abs(rabifreq - 1))
    # powerCarnot = powerful[rabi_index, :]
    energyCarnot = conductionE[rabi_index, :]
    eff='Small System'
    powerful, smoothFid, conductionP, cryoP, energyTotal, conductionE, cryoE = calculate_power_noise(eff, tqb, rabifreq)
    # powerSmall = powerful[rabi_index, :]
    energySmall = conductionE[rabi_index, :]
    # plot_power_vs_temperature(tqb, powerCarnot, powerSmall, 'power_vs_temperature.pdf')

        # Extract energy values at Rabi frequency of 1 MHz
    
    plot_energy_vs_temperature(tqb, energyCarnot, energySmall, 'energy_vs_temperature_conduction_emptyCryo.pdf')
    plot_energy_vs_temperature(tqb, energyCarnot, energySmall, 'energy_vs_temperature_conduction_emptyCryo.svg')

    save_energy_data_to_csv(tqb, energyCarnot, energySmall, 'energy_vs_temperature_conduction_emptyCryo.csv')





if __name__ == "__main__":
    main()
