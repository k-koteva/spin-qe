import copy
import math

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from spin_qe.device.spin_qubitsGlobal import SpinQubit

#import wire_conductivity_JH as wire

hbar = 6.62607004e-34/(2*np.pi)
kb = 1.38064852e-23
g_mu = 2*9.274e-24
w = (1e-3*1.602176634e-19)/hbar #confinement potential energy
m_w = 9.10938356e-31*(w**2) # m*w**2
ec = 1.60217662e-19 # electron charge
antenna_dist = 100e-9 # distance of antenna to spin
grad_B = 1e-3/1e-9
vacuum_imped = 376.730313668
heat_gen = 5e-3


pauli = [np.eye(2),np.array([[0,1],[1,0]]),
        np.array([[0,-1.0j],[1.0j,0]]), np.array([[1,0],[0,-1]])]

def action(op,acted):
    # returns the action of op (list of kraus) on the acted (state)
    if type(op) is list:
        A = np.dot( np.dot(op[0],acted),op[0].conj().T )
        for i in range(1,len(op)):
            A = A + np.dot( np.dot(op[i],acted),op[i].conj().T )
        return A
    else:
        return np.dot( np.dot(op,acted),op.conj().T )

class sc_qubit:
    def __init__(self,temp):
        self.Tq = temp
        self.rabiFreq = np.pi/(25e-9)
        self.A = 10**(5)
        self.level_diff = 6e9/(2*np.pi)

        self.T1time = 80e-6

    def b_wt(self,T):
        # bose-einstein population at temperature T
        return 1.0/(np.exp(hbar*(self.level_diff*2*np.pi)/(kb*T)) -1.0)

    def Gamma1(self):
        ntot = self.b_wt(self.Tq) + self.b_wt(300)/self.A
        g = (1+ntot)*(1.0/self.T1time)
        return g

    def single_infid(self):
        ntot = self.b_wt(self.Tq) + self.b_wt(300)/self.A
        ifd = (1+ntot)*(1.0/self.T1time)/(self.rabiFreq/np.pi)
        return ifd

    def power_cryo(self):
        pq = self.A*(self.rabiFreq**2)*(self.T1time)**2
        pq = ((300 - self.Tq)/self.Tq)*pq

        # give in units of ( gamma_1 *hbar*level_diff  )
        return pq



# things we optimize:
# temperature of qubits, and Rabi Frequency
spin_qubit_params = {
    'n_q': 1,
    'Tq': 0.04,
    'f': 11.20,
    'rabi_in_MHz': 0.5e6,
    'rabi': 0.5,
    'atts_list': [3, 0],
    'stages_ts': [4, 300],
    'silicon_abs': 0.0
}
spin_qubit = SpinQubit(**spin_qubit_params)
# 40 mK, 2 microsec Rabi freq
p_cryo = spin_qubit.total_power()

newSCQC = sc_qubit(40e-3)
print("Typical fidelity for SC qubit: {}".format(1.0 - newSCQC.single_infid()))


def noise_calc(tqb,rabi):
    noiseGrid = np.zeros( (len(tqb),len(rabi)) )
    noiseG2 = np.zeros(len(tqb))

    for i in range(len(tqb)):
        newSCQC = sc_qubit(tqb[i])
        n_val2 = np.min([1.0,newSCQC.single_infid()])
        noiseG2[i] = 1.0- n_val2

        for j in range(len(rabi)):

            spin_qubit_params = {
                'n_q': 1,
                'Tq': tqb[i],
                'f': 11.20,
                'rabi_in_MHz': rabi[j],
                'rabi': rabi[j]*1e-6,
                'atts_list': [3, 0],
                'stages_ts': [4, 300],
                'silicon_abs': 0.0
            }
            spin_qubit = SpinQubit(**spin_qubit_params)
            single_infid = 1 - spin_qubit.fid_1q()
            n_val = np.min([1.0,single_infid])
            noiseGrid[i,j] = 1.0 - n_val

    return noiseGrid,noiseG2

def power_calc(tqb,rabi,type):
    powerGrid = np.zeros( (len(tqb),len(rabi)) )
    powerG2 = np.zeros( len(tqb) )
    
    for i in range(len(tqb)):
        newSCQC = sc_qubit(tqb[i])
        powerG2[i] = newSCQC.power_cryo()

        for j in range(len(rabi)):
            spin_qubit_params = {
                'n_q': 1,
                'Tq': tqb[i],
                'f': 11.20,
                'rabi_in_MHz': rabi[j],
                'rabi': rabi[j]*1e-6,
                'atts_list': [3, 0],
                'stages_ts': [4, 300],
                'silicon_abs': 0.0
            }
            spin_qubit = SpinQubit(**spin_qubit_params)
            powerGrid[i,j] = spin_qubit.total_power()

    return powerGrid,powerG2




tqb = np.linspace(0.02,1,150)
rabi = np.linspace(0.5e6,14e6,120)
T,R = np.meshgrid(tqb,rabi,indexing='ij')

noisy,noiseSC = noise_calc(tqb,rabi)
powerful,powerSC = power_calc(tqb,rabi,type="cryo")

metric_target = np.linspace(0.90,0.985,20)
metric_target_sc = np.linspace(0.975,0.999,20)
opt_power = 1e28*np.ones(len(metric_target))
opt_power_sc = 1e28*np.ones(len(metric_target))
gate_efficiency = np.zeros(len(metric_target))
gate_efficiency_sc = np.zeros(len(metric_target))

for k in range(len(metric_target)):
    for i in range(len(tqb)):
        if noiseSC[i] >= metric_target_sc[k]:
            if opt_power_sc[k] > powerSC[i]:
                opt_power_sc[k] = powerSC[i]
        for j in range(len(rabi)):
            if noisy[i,j] >= metric_target[k]:
                if opt_power[k] > powerful[i,j]:
                    opt_power[k] = powerful[i,j]
    if opt_power[k] == 1e28:
        opt_power[k] = np.nan
    else:
        gate_efficiency[k] = metric_target[k]/opt_power[k]

    if opt_power_sc[k] != 1e28:
        gate_efficiency_sc[k] = metric_target[k]/opt_power_sc[k]

print("\n")
print("Max gate efficiency for SC: {}".format(np.max(gate_efficiency_sc)) )
######### string labels
levels = [1e18,1e19,1e20,1e21]
strs = []
for l in levels:
    l1 = str(l).split('e+')
    strs.append(r'${}\times 10^{{ {} }}$'.format(*l1))
fmt = {l:s for l,s in zip(levels,strs)}

xtl = [20e-3,500e-3,1000e-3,2000e-3,4000e-3]
ytl = [i*1e6 for i in range(5,45,5)]
ytl_strs = []
for i in ytl:
    l1 = str(i*1e-6)
    ytl_strs.append(l1)

mtl = [0.96,0.97,0.98,0.99, 0.995]
################   Each comment block below generates one figure
fig = plt.figure()
ax0 = fig.gca()
noisePlot = ax0.pcolor(T,R,noisy,shading="auto"
                                  )
powerContour = ax0.contour(T,R,powerful
                ,levels
                ,norm=LogNorm(vmin=np.nanmin(powerful),
                              vmax=np.nanmax(powerful) )
                ,colors="red"
                )
clabel = ax0.clabel(powerContour,levels=levels
        ,fmt=fmt
        ,fontsize=10, inline=True)
cbar = fig.colorbar(noisePlot,ax=ax0)
ax0.set_xticks(xtl)
ax0.set_xticklabels([str(i) for i in xtl],fontsize=14)
ax0.set_yticks(ytl)
ax0.set_yticklabels(ytl_strs,fontsize=14)
ax0.set_xlabel('Temperature of quantum chip /K',fontsize=14)
ax0.set_ylabel('Rabi frequency /MHz',fontsize=14)
cbar.ax.set_ylabel('Gate fidelity',fontsize=12)

###########
fig1 = plt.figure()
ax1 = fig1.gca()
ax1.semilogy(metric_target,gate_efficiency,'.k',markersize=15)

ax2 = inset_axes(ax1,width="80%", height="80%",loc=4,
                bbox_to_anchor=(.03, .08, .5, .5),
                bbox_transform=ax1.transAxes
                )
ax2.semilogy(metric_target_sc,gate_efficiency_sc,'.b',markersize=10)
ax2.tick_params(axis='both',which='major',direction='in',length=6,
                top =1, right=1)
ax1.tick_params(axis='both',which='major',direction='in',length=6,
                top =1, right=1)
ax1.tick_params(axis='both',which='minor',direction='in',length=3,top=1,right=1)
# ax1.set_ylim([1e-21,1e-19])
ax1.set_xticks(mtl)
ax1.set_xticklabels([str(i) for i in mtl],fontsize=14)
ax1.tick_params(axis='y',labelsize=14)
ax1.set_xlabel('Gate fidelity', fontsize=14)
ax1.set_ylabel('Gate energetic efficiency',fontsize=14)


###########

x = np.linspace(0,10,80)
y = [i/(1+i**2) for i in x]

fig2 = plt.figure()
ax3 = fig2.gca()
ax3.plot(x,y)

ax3.set_ylabel('Q (in arb. units)',fontsize=14)
ax3.set_xlabel('Rabi frequency (in arb. units)',fontsize=14)
ax3.tick_params(axis='y',labelsize=12)
ax3.tick_params(axis='x',labelsize=12)



# fig.savefig('spinQubit_param.pdf',bbox_inches='tight')
fig1.savefig('twospinQubit_gateEfficiency.pdf',bbox_inches='tight')
# fig2.savefig('T2Rabi_illus.pdf',bbox_inches='tight')

# plt.draw()
# plt.show()




