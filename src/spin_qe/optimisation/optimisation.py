from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from matplotlib.colors import LogNorm

import spin_qe.device.spin_qubits as sq


def calculate_power_noise(tqb: np.ndarray, rabifreq: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    attens = [0, 0, 0, 3, 0, 0]
    stage_ts = [0.007, 0.1, 0.8, 4, 50, 300]
    silicon_abs = 0.01

    powerGrid = np.zeros((len(rabifreq), len(tqb)))
    cryoGrid = np.zeros((len(rabifreq), len(tqb)))
    conductionGrid = np.zeros((len(rabifreq), len(tqb)))
    fidelityGrid = np.zeros((len(rabifreq), len(tqb)))

    for i, rabi in enumerate(rabifreq):
        logger.info(f'Rabi frequency = {rabi}')
        for j, tq in enumerate(tqb):
            logger.info(f'Temperature = {tq}')
            mySQ = sq.SpinQubit(n_q=1, Tq=tq, rabi=rabi, rabi_in_MHz=rabi * 1e6, atts_list=attens, efficiency='Carnot',
                                stages_ts=stage_ts, silicon_abs=silicon_abs)
            powerGrid[i, j] = mySQ.total_power()
            cryoGrid[i, j] = mySQ.cryo_power()
            conductionGrid[i, j] = mySQ.cables_power()
            powerGrid[i, j] = cryoGrid[i, j] + conductionGrid[i, j]
            fidelityGrid[i, j] = mySQ.fid_1q()
            logger.info(f'fid = {mySQ.fid_1q()}')

    return powerGrid, fidelityGrid, conductionGrid, cryoGrid

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

def plot_results(R: np.ndarray, T: np.ndarray, powerful: np.ndarray, smoothFid: np.ndarray, fid_levels: List[float], power_levels: List[float], filename: str):
    mkl_fid = {level: f'${level}$' for level in fid_levels}
    mkl_power = {level: f'${level}$' for level in power_levels}

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
    cbar.ax.set_ylabel('Power /W', fontsize=12)
    fig.savefig(filename, bbox_inches='tight')

def main():
    tqb = np.linspace(0.01, 1, 10)
    rabifreq = np.linspace(0.01, 15, 10)
    R, T = np.meshgrid(rabifreq, tqb, indexing='ij')
    
    powerful, smoothFid, conductionP, cryoP = calculate_power_noise(tqb, rabifreq)
    
    metric_target = np.linspace(0.90, 0.999, 30)
    opt_power, gate_efficiency = optimize_power_and_efficiency(powerful, smoothFid, metric_target)
    
    fid_levels: List[float] = [0.90, 0.99, 0.997]
    power_levels: List[float] = [1, 10, 30]
    plot_results(R, T, powerful, smoothFid, fid_levels, power_levels, 'spinQubit_optimPowSiAbs001realTRIAL_checked.pdf')
    plot_results(R, T, conductionP, smoothFid, fid_levels, power_levels, 'spinQubit_optimPowSiAbs001realTRIAL_checked_conduction.pdf')
    plot_results(R, T, cryoP, smoothFid, fid_levels, power_levels, 'spinQubit_optimPowSiAbs001realTRIAL_checked_cryo.pdf')

if __name__ == "__main__":
    main()
