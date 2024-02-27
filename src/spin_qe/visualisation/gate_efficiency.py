import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


class sc_qubit:
    def __init__(self, temp):
        self.Tq = temp
        self.rabiFreq = np.pi / (25e-9)
        self.A = 10**(5)
        self.level_diff = 6e9 / (2 * np.pi)
        self.T1time = 80e-6

    def b_wt(self, T):
        return 1.0 / (np.exp(hbar * (self.level_diff * 2 * np.pi) / (kb * T)) - 1.0)

    def power_cryo(self):
        pq = self.A * (self.rabiFreq**2) * (self.T1time)**2
        pq = ((300 - self.Tq) / self.Tq) * pq
        return pq

# Assuming the necessary imports and constants are defined elsewhere in your script

# This block calculates gate efficiencies
metric_target = np.linspace(0.90, 0.985, 20)
metric_target_sc = np.linspace(0.975, 0.999, 20)
opt_power = 1e28 * np.ones(len(metric_target))
opt_power_sc = 1e28 * np.ones(len(metric_target))
gate_efficiency = np.zeros(len(metric_target))
gate_efficiency_sc = np.zeros(len(metric_target))

# The loop for calculating efficiencies would go here, assuming it uses `sc_qubit` and other necessary computations

# Plotting the efficiency
fig1 = plt.figure()
ax1 = fig1.gca()
ax1.semilogy(metric_target, gate_efficiency, '.k', markersize=15)

ax2 = inset_axes(ax1, width="80%", height="80%", loc=4, bbox_to_anchor=(.03, .08, .5, .5), bbox_transform=ax1.transAxes)
ax2.semilogy(metric_target_sc, gate_efficiency_sc, '.b', markersize=10)
ax2.tick_params(axis='both', which='major', direction='in', length=6, top=1, right=1)
ax1.tick_params(axis='both', which='major', direction='in', length=6, top=1, right=1)
ax1.tick_params(axis='both', which='minor', direction='in', length=3, top=1, right=1)
ax1.set_xticks(metric_target)
ax1.set_xticklabels([str(i) for i in metric_target], fontsize=14)
ax1.tick_params(axis='y', labelsize=14)
ax1.set_xlabel('Gate fidelity', fontsize=14)
ax1.set_ylabel('Gate energetic efficiency', fontsize=14)

fig1.savefig('twospinQubit_gateEfficiency.pdf', bbox_inches='tight')
