import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from numpy.polynomial import Polynomial
from typing import List

def plot_results(R: np.ndarray, T: np.ndarray, fid_model1: np.ndarray, fid_model2: np.ndarray, filename: str):
    dropbox_folder = os.path.expanduser('~/Dropbox/Apps/Overleaf/QEIW Abstract')
    os.makedirs(dropbox_folder, exist_ok=True)
    
    results_folder = os.path.join(os.getcwd(), 'results')
    os.makedirs(results_folder, exist_ok=True)

    fid_levels_model1 = [0.997, 0.999]
    fid_levels_model2 = [0.997, 0.999]

    fig, ax0 = plt.subplots()
    
    fid_contour_model1 = ax0.contour(R, T, fid_model1, fid_levels_model1, colors="blue", linewidths=1)
    fid_contour_model2 = ax0.contour(R, T, fid_model2, fid_levels_model2, colors="red", linewidths=1)

    ax0.clabel(fid_contour_model1, levels=fid_levels_model1, fmt={level: f'{level}' for level in fid_levels_model1}, fontsize=9, inline=True)
    ax0.clabel(fid_contour_model2, levels=fid_levels_model2, fmt={level: f'{level}' for level in fid_levels_model2}, fontsize=9, inline=True)
    
    ax0.set_yscale('log')
    ax0.set_xscale('log')
    ax0.set_ylabel('Temperature of quantum chip (K)', fontsize=15)
    ax0.set_xlabel('Rabi frequency (MHz)', fontsize=15)

    h1,_ = fid_contour_model1.legend_elements()
    h2,_ = fid_contour_model2.legend_elements()

    ax0.legend([h1[0], h2[0]], ['Fidelity Model 1', 'Fidelity Model 2'], loc='upper left', fontsize=8, ncol=1, bbox_to_anchor=(0.05, 1))
    ax0.tick_params(axis='both', which='major', top=True, right=True,  direction='in', length=8, width=1.5, labelsize=12)
    ax0.tick_params(axis='both', which='minor', top=True, right=True, direction='in', length=4, width=1, labelsize=10)

    fig.savefig(os.path.join(dropbox_folder, filename + '.svg'), bbox_inches='tight')
    fig.savefig(os.path.join(dropbox_folder, filename + '.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(results_folder, filename + '.pdf'), bbox_inches='tight')

def fidelity_model1(Tq, rabi):
    poly_coeff = np.array([5.09137534e+21, -2.77520845e+21, 1.40261875e+22, -1.85030980e+22, 1.02776058e+22])
    my_poly = Polynomial(poly_coeff)
    return 1 - (my_poly(Tq) / rabi**4)

def fidelity_model2(Tq, rabi):
    poly_coeff = np.array([1705.09129852, -929.4116949, 4697.34182489, -6196.65008002, 3441.94938684])
    my_poly = Polynomial(poly_coeff)
    return 1 - (my_poly(Tq) / rabi)

def main():
    tqb = np.logspace(np.log10(0.06), np.log10(10), 100)
    rabifreq = np.logspace(np.log10(0.01), np.log10(100), 100)

    R, T = np.meshgrid(rabifreq, tqb, indexing='ij')

    fid_model1 = fidelity_model1(T, R)
    fid_model2 = fidelity_model2(T, R)
    
    plot_results(R, T, fid_model1, fid_model2, 'fidelity_comparison')

if __name__ == "__main__":
    main()