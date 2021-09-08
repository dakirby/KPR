import matplotlib.pyplot as plt
import numpy as np

from adaptive_sorting import model as as_model
from allosteric import model as allo_model
from dimeric import model as dimeric_model
from homodimeric import model as homodimeric_model
from trimeric import model as trimeric_model
from KPR1 import model as KPR1_model

from pysb_methods import time_course


if __name__ == '__main__':
    test_c = 1E-14*1E-5*6.022E23  # 0.01 pM ligand in 10 uL volume
    test_koff = 1 / 2.  # 2 second bound time, typical of antigen on TCR
    t_end = 10
    dt = 0.1
    num_traj = 11

    # --------------------------------------------------------------------------
    # Notes:
    # - may need to choose test_c for each model so that all models have the
    #   same average 'Cn' at long time
    nT = int(t_end / dt)
    fig, ax = plt.subplots()
    labels = ['Allosteric', 'Homodimeric', 'KPR 1']
    for idx, model in enumerate([allo_model, homodimeric_model, KPR1_model]):
        if labels[idx] == 'Homodimeric':
            test_c = test_c * 1E5
        t = np.linspace(0, t_end, nT)
        y = time_course(model, t_end, num_traj, num_times=nT, params={'L_0': test_c})
        sq_mean_traj = np.square(np.mean(y['Cn'], axis=0))
        var_traj = np.var(y['Cn'], axis=0)
        precision = sq_mean_traj[1:] / var_traj[1:]
        plt.plot(t[1:]*dt, np.log10(precision), label=labels[idx])
        if labels[idx] == 'Homodimeric':
            test_c = test_c / 1E5
    ax.set_xlabel('time (s)')
    ax.set_ylabel('Log Precision')
    ax.legend()
    plt.show()
