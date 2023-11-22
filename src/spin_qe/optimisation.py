import numpy as np

# Assuming spin_qe is the module where SpinQubit is defined
import spin_qe.spin_qubits as sq


def powerNoise_calcSQRab(tqb, rabifreq):
    attens = [3, 0]
    stage_ts = [4, 300]
    silicon_abs = 0.01

    powerGrid = np.zeros((len(rabifreq), len(tqb)))
    fidelityGrid = np.zeros((len(rabifreq), len(tqb)))

    for i, rabi in enumerate(rabifreq):
        for j, tq in enumerate(tqb):
            mySQ = sq.SpinQubit(Tq=tq, rabi=rabi, atts_list=attens,
                                stages_ts=stage_ts, silicon_abs=silicon_abs)
            # Using total_power with PowPi as input
            powerGrid[i, j] = mySQ.total_power()
            # Using fid_1q method for fidelity
            fidelityGrid[i, j] = mySQ.fid_1q()

    return powerGrid, fidelityGrid
