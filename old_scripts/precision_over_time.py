import matplotlib.pyplot as plt
import numpy as np

from adaptive_sorting import model as as_model
from allosteric import model as allo_model
from dimeric import model as dimeric_model
from homodimeric import model as homodimeric_model
from trimeric import model as trimeric_model
from KPR1 import model as KPR1_model

from pysb_methods import time_course


def plot_precision(test_c, t_end, dt):
    """ Plots the ratio <Cn>^2 / Var(Cn) which is the squared SNR of receptor occupancy """
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
        plt.plot(t[1:]*dt, precision, label=labels[idx])
        if labels[idx] == 'Homodimeric':
            test_c = test_c / 1E5
    ax.set_xlabel('time (s)')
    ax.set_ylabel(r'Log $\frac{\langle C_n \rangle^2}{Var(C_n)}$')
    ax.set_yscale('log')
    ax.legend()
    plt.show()


def plot_error(test_c, t_end, dt):
    """ Plots the ratio <Cn> / (<Cn> + <Cw>) which is the H.N. error rate """
    nT = int(t_end / dt)
    fig, ax = plt.subplots()
    labels = ['Allosteric', 'Homodimeric', 'KPR 1']
    for idx, model in enumerate([allo_model, homodimeric_model, KPR1_model]):
        if labels[idx] == 'Homodimeric':
            test_c = test_c * 1E5
        t = np.linspace(0, t_end, nT)
        p = {'L_0': test_c, 'L_self_0': test_c * 10}
        y = time_course(model, t_end, num_traj, num_times=nT, params=p)
        mean_r = np.mean(y['Cn'], axis=0)
        mean_w = np.mean(y['Cw'], axis=0)
        error = mean_r / (mean_r + mean_w)
        plt.plot(t*dt, error, label=labels[idx])
        if labels[idx] == 'Homodimeric':
            test_c = test_c / 1E5
    ax.set_xlabel('time (s)')
    ax.set_ylabel(r'$\eta$ = $\frac{\langle C_n \rangle}{\langle C_n \rangle + \langle C_w \rangle}$')
    ax.set_ylim([0, 1])
    ax.legend()
    plt.show()



if __name__ == '__main__':
    # Notes:
    # - may need to choose test_c for each model so that all models have the
    #   same average 'Cn' at long time
    test_c = 1E-14*1E-5*6.022E23  # 0.01 pM ligand in 10 uL volume
    test_koff = 1 / 2.  # 2 second bound time, typical of antigen on TCR
    t_end = 3000
    dt = 0.1
    num_traj = 30

    plot_directive = 'error'  # ['precision', 'error']
    # --------------------------------------------------------------------------

    if plot_directive == 'precision':
        plot_precision(test_c, t_end, dt)
    elif plot_directive == 'error':
        plot_error(test_c, t_end, dt)
    else:
        print('DID NOT RECOGNIZE PLOT DIRECTIVE')
