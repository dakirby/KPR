import matplotlib.pyplot as plt
import numpy as np

from adaptive_sorting import model as as_model
from allosteric import model as allo_model
from dimeric import model as dimeric_model
from trimeric import model as trimeric_model
from KPR1 import model as KPR1_model

from pysb_methods import dose_response, response_curve


if __name__ == '__main__':
    model_type = 'KPR1'
    plot_dr = True
    plot_rc = False
    crange = np.logspace(0, 4, 15) * 1E-12*1E-5*6.022E23
    koffrange = 1 / np.arange(3, 20, 2)
    t_end = 40
    num_traj = 50

    # --------------------------------------------------------------------------
    if model_type == 'allosteric':
        model = allo_model
        n_threshold = 2E2
        koff_name = 'koff'
    elif model_type == 'KPR1':
        model = KPR1_model
        n_threshold = 100.
        koff_name = 'koff'
        crange = crange * 1E-4
    elif model_type == 'adaptive_sorting':
        model = as_model
        n_threshold = 1.
        koff_name = 'koff'
    elif model_type == 'dimeric':
        model = dimeric_model
        n_threshold = 2E2
        koff_name = 'kd4'  # DOES NOT MAINTAIN DETAILED BALANCE
        crange = 100 * crange
    elif model_type == 'trimeric':
        model = trimeric_model
        n_threshold = 2E2
        koff_name = 'kd4'  # DOES NOT MAINTAIN DETAILED BALANCE
        crange = 100 * crange
    else:
        raise NotImplementedError

    if plot_dr:
        y = dose_response(model, crange, 'L_0', t_end, num_traj)
        mean_traj = np.mean(y['Cn'], axis=0)
        std_traj = np.std(y['Cn'], axis=0)

        fig, ax = plt.subplots()
        plt.plot(crange, mean_traj, 'k--')
        ax.fill_between(crange, mean_traj + std_traj, mean_traj - std_traj, 'k', alpha=0.1)
        plt.xscale('log')
        plt.show()

    if plot_rc:
        response = response_curve(model, crange, 'L_0', koffrange, koff_name, n_threshold, 'Cn', t_end, num_traj)
        fig, ax = plt.subplots()
        plt.plot(1 / koffrange, response, 'k--')
        plt.yscale('log')
        plt.xlabel(r'$\tau$ (s)')
        plt.ylabel('[Ligand]')
        plt.show()
