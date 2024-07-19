import os
from typing import Any, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from matplotlib.colors import LogNorm

import spin_qe.device.spin_qubits as sq


def calculate_power_noise(efficiency: str, tqb: np.ndarray, rabifreq: np.ndarray, calculate_energy: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    attens = [0, 0, 0, 3, 0, 0]
    stage_ts = [0.007, 0.1, 0.8, 4, 50, 300]
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
            conductionGrid[i, j] = mySQ.cables_power()
            powerGrid[i, j] = cryoGrid[i, j] + conductionGrid[i, j]

            energyGrid[i, j] = powerGrid[i, j]*mySQ.gate_t
            cryoEGrid[i, j] = cryoGrid[i, j]*mySQ.gate_t
            conductionEGrid[i, j] = conductionGrid[i, j]*mySQ.gate_t

            fidelityGrid[i, j] = mySQ.fid_1q()
            logger.info(f'fid = {mySQ.fid_1q()}')

    return powerGrid, fidelityGrid, conductionGrid, cryoGrid, energyGrid, conductionEGrid, cryoEGrid 

def calculate_noise(efficiency: str, tqb: np.ndarray, rabifreq: np.ndarray, calculate_energy: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    attens = [0, 0, 0, 3, 0, 0]
    stage_ts = [0.007, 0.1, 0.8, 4, 50, 300]
    silicon_abs = 0.0

    fidelityModel1 = np.zeros((len(rabifreq), len(tqb)))
    fidelityModel2 = np.zeros((len(rabifreq), len(tqb)))

    for i, rabi in enumerate(rabifreq):
        logger.info(f'Rabi frequency = {rabi}')
        for j, tq in enumerate(tqb):
            logger.info(f'Temperature = {tq}')
            mySQ = sq.SpinQubit(n_q=1, Tq=tq, rabi=rabi, rabi_in_MHz=rabi * 1e6, atts_list=attens, efficiency=efficiency,
                                stages_ts=stage_ts, silicon_abs=silicon_abs)
            fidelityModel1[i, j] = mySQ.fidelity(model='Model 1')
            fidelityModel2[i, j] = mySQ.fidelity(model='Model 2')

    return fidelityModel1, fidelityModel2

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

def plot_fidelity(R: np.ndarray, T: np.ndarray, fid_model1: np.ndarray, fid_model2: np.ndarray, fid_levels: List[float], filename: str):
            # Define the path to the Results folder
    # dropbox_folder = os.path.expanduser('~/Dropbox/Apps/Overleaf/Energetics NISQ Spin qubits/Results')
    dropbox_folder = os.path.expanduser('~/Dropbox/Apps/Overleaf/QEIW Abstract')
    
    # Create the Results folder if it doesn't exist
    os.makedirs(dropbox_folder, exist_ok=True)
    
    # Define the path to the results folder in the current directory
    results_folder = os.path.join(os.getcwd(), 'results')
    
    # Create the local results folder if it doesn't exist
    os.makedirs(results_folder, exist_ok=True)
    fid_model1 = np.where(fid_model1 <= 0, 1e-10, fid_model1)
    fid_model2 = np.where(fid_model2 <= 0, 1e-10, fid_model2)
    mkl_fid = {level: f'${level}$' for level in fid_levels}
    fig = plt.figure()
    ax0 = fig.gca()
    plt.xscale('log')
    plt.yscale('log')
    fid_contour1 = ax0.contour(R, T, fid_model1, fid_levels,
                              norm=LogNorm(vmin=float(np.nanmin(fid_model1)),
                                           vmax=float(np.nanmax(fid_model1))),
                              colors="orange", linewidths=1)
    

    fid_contour2 = ax0.contour(R, T, fid_model2, fid_levels,
                              norm=LogNorm(vmin=float(np.nanmin(fid_model2)),
                                           vmax=float(np.nanmax(fid_model2))),
                              colors="blue", linewidths=1)
    
    ax0.clabel(fid_contour1, levels=fid_levels, fmt=mkl_fid, fontsize=9, inline=True)
    ax0.clabel(fid_contour2, levels=fid_levels, fmt=mkl_fid, fontsize=9, inline=True)

    ax0.set_ylabel('Temperature of quantum chip (K)', fontsize=15)
    ax0.set_xlabel('Rabi frequency (MHz)', fontsize=15)


    h1,_ = fid_contour1.legend_elements()
    h2,_ = fid_contour2.legend_elements()
    ax0.legend([h1[0], h2[0]], ['Model1', 'Model2'], loc='upper left', fontsize=12, ncol=1, bbox_to_anchor=(0.05, 1))
    ax0.tick_params(axis='both', which='major', top=True, right=True,  direction='in', length=8, width=1.5, labelsize=12)
    ax0.tick_params(axis='both', which='minor', top=True, right=True, direction='in', length=4, width=1, labelsize=10)
    fig.savefig(os.path.join(dropbox_folder, filename + '.pdf'), bbox_inches='tight')
    
    # Save the plot as PDF in local results folder
    fig.savefig(os.path.join(results_folder, filename + '.pdf'), bbox_inches='tight')


def plot_results(R: np.ndarray, T: np.ndarray, powerful: np.ndarray, smoothFid: np.ndarray, fid_levels: List[float], power_levels: List[float], filename: str, plot_energy: bool = False):
        # Define the path to the Results folder
    # dropbox_folder = os.path.expanduser('~/Dropbox/Apps/Overleaf/Energetics NISQ Spin qubits/Results')
    dropbox_folder = os.path.expanduser('~/Dropbox/Apps/Overleaf/QEIW Abstract')
    
    # Create the Results folder if it doesn't exist
    os.makedirs(dropbox_folder, exist_ok=True)
    
    # Define the path to the results folder in the current directory
    results_folder = os.path.join(os.getcwd(), 'results')
    
    # Create the local results folder if it doesn't exist
    os.makedirs(results_folder, exist_ok=True)

    mkl_fid = {level: f'${level}$' for level in fid_levels}
    mkl_power = {level: f'${level}$' for level in power_levels}
    label = 'Energy per gate (J)' if plot_energy else 'Power /W'
    fig = plt.figure()
    ax0 = fig.gca()
    powerPlot = ax0.pcolor(R, T, powerful, 
                           norm=LogNorm(vmin=float(np.nanmin(powerful)), vmax=float(np.nanmax(powerful))),
                           shading="auto")

    smoothFid = np.where(smoothFid <= 0, 1e-10, smoothFid)
    
    fid_contour = ax0.contour(R, T, smoothFid, fid_levels,
                              norm=LogNorm(vmin=float(np.nanmin(smoothFid)) + 1e-10,
                                           vmax=float(np.nanmax(smoothFid)) + 1e-10),
                              colors="black", linewidths=1)

    power_contour = ax0.contour(R, T, powerful, power_levels,
                                norm=LogNorm(vmin=float(np.nanmin(powerful)),
                                             vmax=float(np.nanmax(powerful))),
                                colors="white", linewidths=1)

    ax0.clabel(fid_contour, levels=fid_levels, fmt=mkl_fid, fontsize=9, inline=True)
    ax0.clabel(power_contour, levels=power_levels, fmt=mkl_power, fontsize=9, inline=True)
    # ax0.set_title('Driving 1 qubit')
    ax0.set_yscale('log')
    ax0.set_xscale('log')
    cbar = fig.colorbar(powerPlot, ax=ax0)
    ax0.set_ylabel('Temperature of quantum chip (K)', fontsize=15)
    ax0.set_xlabel('Rabi frequency (MHz)', fontsize=15)

    cbar.ax.set_ylabel(label, fontsize=15)
    cbar.ax.tick_params(axis='both', which='major', length=8, width=1.5, labelsize=12)
    cbar.ax.tick_params(axis='both', which='minor', length=4, width=1, labelsize=10)

    h1,_ = fid_contour.legend_elements()
    h2,_ = power_contour.legend_elements()

    ax0.legend([h1[0], h2[0]], ['Gate\nfidelity', 'Energy\nper\ngate'], loc='upper left', fontsize=8, ncol=1, bbox_to_anchor=(0.05, 1))
    ax0.tick_params(axis='both', which='major', top=True, right=True,  direction='in', length=8, width=1.5, labelsize=12)
    ax0.tick_params(axis='both', which='minor', top=True, right=True, direction='in', length=4, width=1, labelsize=10)
        # Save the plot as SVG in Dropbox folder
    fig.savefig(os.path.join(dropbox_folder, filename + '.svg'), bbox_inches='tight')
    fig.savefig(os.path.join(dropbox_folder, filename + '.pdf'), bbox_inches='tight')
    
    # Save the plot as PDF in local results folder
    fig.savefig(os.path.join(results_folder, filename + '.pdf'), bbox_inches='tight')

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
    dropbox_folder = os.path.expanduser('~/Dropbox/Apps/Overleaf/Energetics NISQ Spin qubits/Results')
    # Create the Results folder if it doesn't exist
    os.makedirs(dropbox_folder, exist_ok=True)
    plt.title('1 Qubit at 10MHz', fontsize=16) 
    plt.plot(tqb, energyCarnot, label='Carnot Efficiency', linewidth=2.5)
    plt.plot(tqb, energySmall, label='Small System Efficiency', linestyle='--', linewidth=2.5)
    plt.xlabel('Temperature of quantum chip /K', fontsize=15)
    plt.xticks(fontsize=17)  # Increased tick size
    plt.yticks(fontsize=17)
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel('Energy from Cryogenics /J', fontsize=15)
    plt.legend(fontsize=15)
    plt.tick_params(axis='both', which='both', direction='in', top=True, right=True, width=2.5)  # Ticks on top and right
    plt.savefig(os.path.join(filename + '.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(dropbox_folder, filename + '.pdf'), bbox_inches='tight')

def plot_energy_vs_rabi(rabi: np.ndarray, energyCarnot: np.ndarray, energySmall: np.ndarray, filename: str):
    dropbox_folder = os.path.expanduser('~/Dropbox/Apps/Overleaf/Energetics NISQ Spin qubits/Results')
    # Create the Results folder if it doesn't exist
    os.makedirs(dropbox_folder, exist_ok=True)
    plt.title('1 Qubit at 10MHz', fontsize=16) 
    plt.plot(rabi, energyCarnot, label='Carnot Efficiency', linewidth=2.5)
    plt.plot(rabi, energySmall, label='Small System Efficiency', linestyle='--', linewidth=2.5)
    plt.xlabel('Rabi frequency /K', fontsize=15)
    plt.xticks(fontsize=17)  # Increased tick size
    plt.yticks(fontsize=17)
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel('Energy from Cryogenics (J)', fontsize=15)
    plt.legend(fontsize=15)
    plt.tick_params(axis='both', which='both', direction='in', top=True, right=True, width=2.5)  # Ticks on top and right
    plt.savefig(os.path.join(filename + '.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(dropbox_folder, filename + '.pdf'), bbox_inches='tight')

def find_min_and_index(values: list) -> Tuple[Any, int]:
    min_value = min(values)
    min_index = values.index(min_value)
    return min_value, min_index

def main():
    nq = 100
    tqb = np.logspace(np.log10(0.06), np.log10(10), 100)
    # rabifreq = np.linspace(0.01, 100, 100)
    rabifreq = np.logspace(np.log10(0.01), np.log10(100), 100)

    R, T = np.meshgrid(rabifreq, tqb, indexing='ij')
    eff = 'Carnot'
    powerful, smoothFid, conductionP, cryoP, energyTotal, conductionE, cryoE = calculate_power_noise(eff, tqb, rabifreq)
    
    metric_target = np.linspace(0.90, 0.999, 30)
    opt_power, gate_efficiency = optimize_power_and_efficiency(powerful, smoothFid, metric_target)
    
    fid_levels: List[float] = [0.9, 0.996, 0.9999]
    power_levels: List[float] = [1, 10, 30]
    energy_levels: List[float] = [1e-8, 1e-7, 1e-6]
    # energy_levels: List[float] = [1e-6, 1e-5, 1e-4]
    # power_levels: List[float] = [1e2, 1e5, 1e7]
    # energy_levels: List[float] = [1e-3, 1e-1]
    # plot_results(R, T, powerful, smoothFid, fid_levels, power_levels, 'carnotspinQubit_optimPowSiAbs001realTRIAL_checked.pdf')
    # plot_results(R, T, conductionP, smoothFid, fid_levels, power_levels, 'carnotspinQubit_optimPowSiAbs001realTRIAL_checked_conduction.pdf')
    # plot_results(R, T, cryoP, smoothFid, fid_levels, power_levels, 'carnotspinQubit_optimPowSiAbs001realTRIAL_checked_cryo.pdf')
    plot_results(R, T, energyTotal, smoothFid, fid_levels, energy_levels, 'total_energy_carnot_1Q', plot_energy=True)
    plot_results(R, T, conductionE, smoothFid, fid_levels, energy_levels, 'conduction_energy_carnot_1Q', plot_energy=True)
    plot_results(R, T, cryoE, smoothFid, fid_levels, energy_levels, 'cryo_energy_carnot_1Q', plot_energy=True)

    #  Extract power values at Rabi frequency of 10 MHz

    rabi_index = np.argmin(np.abs(rabifreq - 10))
    # powerCarnot = powerful[rabi_index, :]
    # energyCarnot = energyTotal[rabi_index, :]
    energyCarnot = cryoE[rabi_index, :]
    
    
    eff='Small System'
    powerful, smoothFid, conductionP, cryoP, energyTotal, conductionE, cryoE = calculate_power_noise(eff, tqb, rabifreq)
    # powerSmall = powerful[rabi_index, :]
    # energySmall = energyTotal[rabi_index, :]
    energySmall = cryoE[rabi_index, :]
    # plot_power_vs_temperature(tqb, powerCarnot, powerSmall, 'power_vs_temperature.pdf')

        # Extract energy values at Rabi frequency of 10 MHz
    
    # plot_energy_vs_temperature(tqb, energyCarnot, energySmall, '1Q_cryogenics_energy_vs_temperature_compare')


    # #Extract energy values at Temp = 700mk
    # temp_index = np.argmin(np.abs(tqb - 10))
    # energyCarnot = cryoE[rabi_index, :]

def main2():
    # rabifreq = np.linspace(0.01, 100, 100)
    # tqb = np.linspace(0.06, 10, 100)
    tqb = np.logspace(np.log10(0.06), np.log10(10), 100)
    rabifreq = np.logspace(np.log10(0.01), np.log10(100), 100)

    R, T = np.meshgrid(rabifreq, tqb, indexing='ij')
    eff = 'Carnot'
    fidModel1, fidModel2 = calculate_noise(eff, tqb, rabifreq)
    fid_levels = [0.997, 0.998]
    plot_fidelity(R, T, fidModel1, fidModel2, fid_levels, 'fidelity_model1and2')


if __name__ == "__main__":
    main2()
