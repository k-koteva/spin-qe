import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

# Assuming spin_qe is the module where SpinQubit is defined
import spin_qe.device.spin_qubits as sq


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
            fidelityGrid[i, j] = mySQ.fid_1q()
            print('fid = ', mySQ.fid_1q())

    return powerGrid, fidelityGrid

tqb = np.linspace(0.01,1,10)
rabifreq = np.linspace(0.01,15,10)
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

mylevels = [0.90, 0.99, 0.997]
mkl = {0.90: '$0.90$', 0.99: '$0.99$',0.997: '$0.997$'}
fig = plt.figure()
ax0 = fig.gca()
powerPlot = ax0.pcolor(R,T,powerful, 
           norm=LogNorm(vmin=float(np.nanmin(powerful)), vmax=float(np.nanmax(powerful))),
           shading="auto")

# noiseContour = ax0.contour(R,T,noisy
#                 ,mylevels
#                 ,norm=LogNorm(vmin=np.nanmin(powerful),
#                               vmax=np.nanmax(powerful) )
#                 ,colors="red"
#                 )
smoothFid = np.where(smoothFid <= 0, 1e-10, smoothFid)
noiseContour = ax0.contour(R,T,smoothFid
                ,mylevels
                ,norm=LogNorm(vmin=float(np.nanmin(smoothFid)) + 1e-10,
                              vmax=float(np.nanmax(smoothFid)) + 1e-10 )
                ,colors="black"
                )

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
fig.savefig('spinQubit_optimPowSiAbs001realTRIAL_check.pdf',bbox_inches='tight')

