import os
from typing import Any, List, Tuple, Union, Optional

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from matplotlib.colors import LogNorm
from scipy.ndimage import minimum_position
from matplotlib import colormaps
from matplotlib.colors import LinearSegmentedColormap
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
            mySQ = sq.SpinQubit(n_q=20, Tq=tq, rabi=rabi, rabi_in_MHz=rabi * 1e6, atts_list=attens, efficiency=efficiency,
                                stages_ts=stage_ts, silicon_abs=silicon_abs)
            cryoGrid[i, j] = mySQ.cryo_power()
            conductionGrid[i, j] = mySQ.cables_power()
            powerGrid[i, j] = mySQ.total_power()
            # powerGrid[i, j] = cryoGrid[i, j] + conductionGrid[i, j]

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
            mySQ = sq.SpinQubit(n_q=20, Tq=tq, rabi=rabi, rabi_in_MHz=rabi * 1e6, atts_list=attens, efficiency=efficiency,
                                stages_ts=stage_ts, silicon_abs=silicon_abs)
            fidelityModel1[i, j] = mySQ.fidelity(model='Slow')
            fidelityModel2[i, j] = mySQ.fidelity(model='Markov')

    return fidelityModel1, fidelityModel2

def calculate_noise_nomeas(efficiency: str, tqb: np.ndarray, rabifreq: np.ndarray, calculate_energy: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    attens = [0, 0, 0, 3, 0, 0]
    stage_ts = [0.007, 0.1, 0.8, 4, 50, 300]
    silicon_abs = 0.0

    fidelityModel1 = np.zeros((len(rabifreq), len(tqb)))
    fidelityModel2 = np.zeros((len(rabifreq), len(tqb)))

    for i, rabi in enumerate(rabifreq):
        logger.info(f'Rabi frequency = {rabi}')
        for j, tq in enumerate(tqb):
            logger.info(f'Temperature = {tq}')
            mySQ = sq.SpinQubit(n_q=20, Tq=tq, rabi=rabi, rabi_in_MHz=rabi * 1e6, atts_list=attens, efficiency=efficiency,
                                stages_ts=stage_ts, silicon_abs=silicon_abs)
            fidelityModel1[i, j] = mySQ.fidelity_nomeas(model='Slow')
            fidelityModel2[i, j] = mySQ.fidelity_nomeas(model='Markov')

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
    dropbox_folder = os.path.expanduser('~/Dropbox/Apps/Overleaf/Energetics NISQ Spin qubits/Results2')
    # dropbox_folder = os.path.expanduser('~/Dropbox/Apps/Overleaf/QEIW Abstract/')
    
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
    dropbox_folder = os.path.expanduser('~/Dropbox/Apps/Overleaf/Energetics NISQ Spin qubits/Results2')
    # dropbox_folder = os.path.expanduser('~/Dropbox/Apps/Overleaf/QEIW Abstract')
    
    # Create the Results folder if it doesn't exist
    os.makedirs(dropbox_folder, exist_ok=True)
    
    # Define the path to the results folder in the current directory
    results_folder = os.path.join(os.getcwd(), 'results')
    
    # Create the local results folder if it doesn't exist
    os.makedirs(results_folder, exist_ok=True)

    mkl_fid = {level: f'${level}$' for level in fid_levels}
    mkl_power = {level: f'${level}$' for level in power_levels}
    label = 'Energy per timestep (J)' if plot_energy else 'Power /W'
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

    ax0.legend([h1[0], h2[0]], ['Gate\nfidelity', 'Energy\nper\ntimestep'], loc='upper left', fontsize=8, ncol=1, bbox_to_anchor=(0.05, 1))
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
    dropbox_folder = os.path.expanduser('~/Dropbox/Apps/Overleaf/Energetics NISQ Spin qubits/Results2')
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
    dropbox_folder = os.path.expanduser('~/Dropbox/Apps/Overleaf/Energetics NISQ Spin qubits/Results2')
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
    # nq = 100
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

def find_optimal_location(fid_model: np.ndarray, energy_total: np.ndarray, fid_level: float) -> Optional[Tuple[Tuple[int, int], float]]:
    """
    Finds the optimal location in fid_model close to fid_level that minimizes energy_total.

    Parameters:
    - fid_model (np.ndarray): The fidelity model array.
    - energy_total (np.ndarray): The energy total array.
    - fid_level (float): The target fidelity level.

    Returns:
    - Tuple[Tuple[int, int], float]: The optimal location (row, col) and the corresponding minimum energy.
      Returns None if no close values to fid_level are found.
    """
    # Get a mask for locations close to the target fid_level
    close_mask = np.isclose(fid_model, fid_level, atol=1e-6)

    # If no values are close to the fid_level, return None
    if not np.any(close_mask):
        return None

    # Extract the energy values for the matching locations
    energy_subset = np.where(close_mask, energy_total, np.inf)

    # Find the location of the minimum energy value within the subset
    min_pos = minimum_position(energy_subset)
    min_energy = energy_subset[min_pos]

    return min_pos, min_energy




def plot_combined_results(R: np.ndarray, T: np.ndarray, powerful: np.ndarray,
                          fid_model1: np.ndarray, fid_model2: np.ndarray,
                          fid_levels: List[float], power_levels: List[float],
                          filename: str, plot_energy: bool = False, min_powers: List[Optional[Tuple[int, int]]] = [None]):
    # Define the path to the Results folder
    dropbox_folder = os.path.expanduser('~/Dropbox/Apps/Overleaf/Energetics NISQ Spin qubits/Results2')
    # Create the Results folder if it doesn't exist
    os.makedirs(dropbox_folder, exist_ok=True)
    
    # Define the path to the results folder in the current directory
    results_folder = os.path.join(os.getcwd(), 'results')
    # Create the local results folder if it doesn't exist
    os.makedirs(results_folder, exist_ok=True)
    
    # Handle negative or zero values in fid_model1 and fid_model2
    fid_model1 = np.where(fid_model1 <= 0, 1e-10, fid_model1)
    fid_model2 = np.where(fid_model2 <= 0, 1e-10, fid_model2)

    # Define the custom colormap
    colors = [
    "#004C40",  # **Deep Petrol Green (Darkest Start)**
    "#005F50",  # **Muted Petrol Green**
    "#00705A",  # **Dark Teal-Green**
    "#008060",  # **Deep Green-Teal**
    "#00916F",  # **Teal-Green Transition**
    "#22A884",  # **True Teal (Viridis Exact)**
    "#35B779",  # **Teal-Green**
    "#48C16E",  # **Mid Green-Teal**
    "#60CA5D",  # **Strong Green**
    "#7CD250",  # **Balanced Bright Green**
    "#99D83F",  # **Lighter Green**
    "#B5DE2B",  # **Green-Yellow (Reduced)**
    "#D0EA10",  # **Yellow-Green**
    "#FFFF00",  # **Yellow (ONLY ONE NOW)**
    "#FFD000",  # **Yellow-Orange**
    "#FFA500",  # **Orange**
    "#FF8700",  # **Deep Orange**
    "#FF7300",  # **Red-Orange**
    "#FF4500",  # **Bloody Orange (Deep Red-Orange)**
    "#D32F2F"   # **Dark Red (Final Deep Touch)**
]
    custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)

    # Use the custom colormap
    cmap = custom_cmap
    

    mkl_fid = {level: f'${level}$' for level in fid_levels}
    mkl_power = {level: f'${level}$' for level in power_levels}
    label = 'Energy per timestep (J)' if plot_energy else 'Power /W'
    
    fig = plt.figure()
    ax0 = fig.gca()
    value = 1e-4 * 10**2
    norm = LogNorm(vmin=1e-8, vmax=value)
    norm = LogNorm()
    # cmap = colormaps['viridis']
    # powerPlot = ax0.pcolor(R, T, powerful, 
    #                        norm=LogNorm(vmin=float(np.nanmin(powerful)), vmax=float(np.nanmax(powerful))),
    #                        shading="auto")

    powerPlot = ax0.pcolor(R, T, powerful, 
                           norm=norm,
                           cmap=cmap,
                           shading="auto")

    # Plot the contours for fid_model1 and fid_model2
    fid_contour1 = ax0.contour(R, T, fid_model1, fid_levels,
                               norm=LogNorm(vmin=float(np.nanmin(fid_model1)),
                                            vmax=float(np.nanmax(fid_model1))),
                               colors="#006CFF", linewidths=2)
    
    fid_contour2 = ax0.contour(R, T, fid_model2, fid_levels,
                               norm=LogNorm(vmin=float(np.nanmin(fid_model2)),
                                            vmax=float(np.nanmax(fid_model2))),
                               colors="#8000FF", linewidths=2)

    power_contour = ax0.contour(R, T, powerful, power_levels,
                                norm=LogNorm(vmin=float(np.nanmin(powerful)),
                                             vmax=float(np.nanmax(powerful))),
                                colors="#404040", linewidths=2)

    # ax0.clabel(fid_contour1, levels=fid_levels, fmt=mkl_fid, fontsize=7, inline=True)
    # ax0.clabel(fid_contour2, levels=fid_levels, fmt=mkl_fid, fontsize=7, inline=True)
    # ax0.clabel(power_contour, levels=power_levels, fmt=mkl_power, fontsize=7, inline=True)

    ax0.set_yscale('log')
    ax0.set_xscale('log')
    cbar = fig.colorbar(powerPlot, ax=ax0)
    ax0.set_ylabel('Temperature of qubits (K)', fontsize=15)
    ax0.set_xlabel('Rabi frequency (MHz)', fontsize=15)

    cbar.ax.set_ylabel(label, fontsize=15)
    cbar.ax.tick_params(axis='both', which='major', length=8, width=1.5, labelsize=12)
    cbar.ax.tick_params(axis='both', which='minor', length=4, width=1, labelsize=10)

    h1, _ = fid_contour1.legend_elements()
    h2, _ = fid_contour2.legend_elements()
    h3, _ = power_contour.legend_elements()


    # R_mesh, T_mesh = np.meshgrid(R, T, indexing='ij')

    # for contour, color, label in [(fid_contour1, 'orange', 'Model1 Min Power'), 
    #                             (fid_contour2, 'blue', 'Model2 Min Power')]:
    #     min_power = float('inf')
    #     min_coords = (0, 0)
    #     for segment in contour.allsegs[0]:  # Process the first contour level
    #         for point in segment:
    #             # Find the closest indices on the meshgrid
    #             r_idx = np.abs(R_mesh - point[0]).argmin()
    #             t_idx = np.abs(T_mesh - point[1]).argmin()
                
    #             # Ensure indices are within bounds
    #             if 0 <= r_idx < powerful.shape[1] and 0 <= t_idx < powerful.shape[0]:
    #                 if powerful[t_idx, r_idx] < min_power:
    #                     min_power = powerful[t_idx, r_idx]
    #                     min_coords = (point[0], point[1])
        
    #     # Plot the point with the lowest power
    #     ax0.plot(min_coords[0], min_coords[1], 'o', color=color, markersize=5, label=label)

        # Plot the stars on specified positions
    if min_powers is not None:
        for position in min_powers:
            if position is not None:  # Ensure the position is valid
                ax0.plot(R[position[0], position[1]], T[position[0], position[1]], marker='*', color='silver', markersize=10, label='Optimal Power')

    ax0.legend([h1[0], h2[0], h3[0]], ['Static noise', 'Markovian noise', 'Energy per timestep'], loc='upper left', fontsize=8, ncol=1, bbox_to_anchor=(0.05, 1))
    ax0.tick_params(axis='both', which='major', top=True, right=True,  direction='in', length=8, width=1.5, labelsize=12)
    ax0.tick_params(axis='both', which='minor', top=True, right=True, direction='in', length=4, width=1, labelsize=10)

    # Save the plot in Dropbox and local results folders
    # fig.savefig(os.path.join(dropbox_folder, filename + '.svg'), bbox_inches='tight')
    # fig.savefig(os.path.join(dropbox_folder, filename + '.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(results_folder, filename + '.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(results_folder, filename + '.svg'), bbox_inches='tight')

def main3():
    # Define temperature and Rabi frequency ranges
    tqb = np.logspace(np.log10(0.06), np.log10(10), 100)
    rabifreq = np.logspace(np.log10(0.01), np.log10(1000), 100)
    R, T = np.meshgrid(rabifreq, tqb, indexing='ij')

    # Calculate power and noise metrics for 'Carnot' efficiency
    eff = 'Small System'
    powerful, _, conductionP, cryoP, energyTotal, conductionE, cryoE = calculate_power_noise(eff, tqb, rabifreq)
    
    # Calculate fidelity models for 'Carnot' efficiency
    fidModel1, fidModel2 = calculate_noise(eff, tqb, rabifreq)

    fidModel3, fidModel4 = calculate_noise_nomeas(eff, tqb, rabifreq)

    # Define contour levels for fidelity and power/energy
    fid_levels = [0.5, 0.8, 0.9]  # Adjust as necessary
    power_levels = [1, 10, 30]
    # energy_levels = [1e-7, 1e-6, 1e-5]
    energy_levels = [1e-4, 1e-3, 1e-2]
    # position1, energy_min = find_optimal_location(fidModel1, energyTotal, 0.995)
    # position2, energy_min = find_optimal_location(fidModel2, energyTotal, 0.995)
    # position3, energy_min = find_optimal_location(fidModel1, energyTotal, 0.99)
    # position4, energy_min = find_optimal_location(fidModel2, energyTotal, 0.99)
    # positions = [position1, position2, position3, position4]

    # Plot combined results for total energy
    plot_combined_results(R, T, energyTotal, fidModel1, fidModel2, fid_levels, energy_levels, 'total_energy_SS_combined20Q1000D_meas', plot_energy=True)
    plot_combined_results(R, T, energyTotal, fidModel3, fidModel4, fid_levels, energy_levels, 'total_energy_SS_combined20Q1000D', plot_energy=True)
    
    # logger.info(f'Minimum energy at {position} with value {energy_min}')


    # # Plot combined results for conduction energy
    # plot_combined_results(R, T, conductionE, fidModel1, fidModel2, fid_levels, energy_levels, 'conduction_energy_carnot_combined', plot_energy=True)

    # # Plot combined results for cryogenics energy
    # plot_combined_results(R, T, cryoE, fidModel1, fidModel2, fid_levels, energy_levels, 'cryo_energy_carnot_combined', plot_energy=True)

    # # Optionally, plot results for power if needed
    # plot_combined_results(R, T, powerful, fidModel1, fidModel2, fid_levels, power_levels, 'power_carnot_combined', plot_energy=False)

# Call the main3 function to generate the plots

if __name__ == "__main__":
    main3()
