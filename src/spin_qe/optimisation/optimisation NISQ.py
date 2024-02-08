import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from matplotlib.colors import LogNorm

# Assuming spin_qe is the module where SpinQubit is defined
import spin_qe.device.spin_qubitsGlobal as sq


def powerNoise_calcSQRab(tqb, rabifreq):
    attens = [3, 0]
    stage_ts = [4, 300]
    silicon_abs = 0.01

    powerGrid = np.zeros((len(rabifreq), len(tqb)))
    fidelityGrid = np.zeros((len(rabifreq), len(tqb)))

    for i, rabi in enumerate(rabifreq):
        print('Rabi frequency = ', rabi)
        for j, tq in enumerate(tqb):
            print('Temperature = ', tq)
            mySQ = sq.SpinQubit(Tq=tq, rabi = rabi, rabi_in_MHz=rabi*1e6, atts_list=attens,
                                stages_ts=stage_ts, silicon_abs=silicon_abs)
            # Using total_power with PowPi as input
            powerGrid[i, j] = mySQ.total_power()
            # Using fid_1q method for fidelity
            # fidelityGrid[i, j] = mySQ.fid_circ(num_1q_gates=6, num_2q_gates=1, num_meas=5)
            fidelityGrid[i, j] = mySQ.fid_1q()
            print('fid = ', mySQ.fid_1q())

    return powerGrid, fidelityGrid

tqb = np.linspace(0.01,1,20)
rabifreq = np.linspace(0.01,15,20)
R,T = np.meshgrid(rabifreq,tqb,indexing='ij')
powerful, smoothFid = powerNoise_calcSQRab(tqb,rabifreq)

metric_target = np.linspace(0.90,0.999,30)
opt_power = 1e28*np.ones(len(metric_target))
gate_efficiency = np.zeros(len(metric_target))

for k in range(len(metric_target)):
    for i in range(len(rabifreq)):
        for j in range(len(tqb)):
            if smoothFid[i,j] >= metric_target[k]:
                if opt_power[k] > powerful[i,j]:
                    opt_power[k] = powerful[i,j]
    if opt_power[k] == 1e28:
        opt_power[k] = np.nan
    else:
        gate_efficiency[k] = metric_target[k]/opt_power[k]

# mylevels = [0.95, 0.99, 0.999]
# mkl = {0.95: '$0.95$', 0.99: '$0.99$',0.999: '$0.999$'}
mylevels = [ 0.9974, 0.99996]
mkl = { 0.9974: '$0.9974$',0.99996: '$0.9996$'}
fig = plt.figure()
ax0 = fig.gca()
powerPlot = ax0.pcolor(R,T,powerful, 
           norm=LogNorm(vmin=float(np.nanmin(powerful)), vmax=float(np.nanmax(powerful))),
           shading="auto")


smoothFid = np.where(smoothFid <= 0, 1e-10, smoothFid)




noiseContour = ax0.contour(R,T,smoothFid
                ,mylevels
                ,norm=LogNorm(vmin=float(np.nanmin(smoothFid)) + 1e-10,
                              vmax=float(np.nanmax(smoothFid)) + 1e-10 )
                ,colors="black"
                )

# paths = noiseContour.collections[0].get_paths()
# print('len(paths):', len(paths))
# print('see path', paths[0])
# # for collection in noiseContour.collections:
# #     collection.remove()

# vertices = paths[0].vertices  # Extract vertices from the path
# # Find the midpoint index of the vertices array
# midpoint_index = len(vertices) // 2

# # Select only the second half of the vertices
# second_half_vertices = vertices[midpoint_index:]

# # Plotting the second half of the path directly on ax0
# ax0.plot(second_half_vertices[:, 0], second_half_vertices[:, 1], color='black', linestyle='-')


clabel = ax0.clabel(noiseContour,levels=mylevels
        ,fmt=mkl 
        ,fontsize=10, inline=True)
# clabel = ax0.clabel(smoothFid,levels=mylevels
#         ,fmt=mkl
#         ,fontsize=10, inline=True)

ax0.set_yscale('log')
# ax0.set_xscale('log')
cbar = fig.colorbar(powerPlot,ax=ax0)

ax0.set_ylabel('Temperature of quantum chip /K',fontsize=14)
ax0.set_xlabel('Rabi frequency /MHz',fontsize=14)
cbar.ax.set_ylabel('Power /W',fontsize=12)
fig.savefig('spinQubit_1qb_new.pdf',bbox_inches='tight')

