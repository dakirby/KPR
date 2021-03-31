import matplotlib.pyplot as plt
import numpy as np
from pysb.simulator.bng import BngSimulator
from pysb import Initial, Monomer
import warnings

from adaptive_sorting import model as as_model
from allosteric import model as allo_model
from dimeric import model as dimeric_model


def dose_response(model, c_range, c_name, t_end, n_runs, num_times=10):
    assert n_runs > 1
    t = np.linspace(0, t_end, num_times)
    observables = model.observables.keys()
    resp = {obs: np.zeros((n_runs, len(c_range))) for obs in observables}
    # simulate
    for c_idx, c in enumerate(c_range):
        model.parameters[c_name].value = c
        sim = BngSimulator(model)
        x = sim.run(tspan=t, verbose=False, n_runs=n_runs, method='ssa')
        y = np.array(x.observables)
        for obs in observables:
            resp[obs][:, c_idx] = y[obs][:, -1]
    return resp


def response_curve(model, c_range, c_name, koff_range, koff_name, obs_thrs, obs_name, t_end, n_runs):
    c_typical = np.zeros(len(koff_range))
    for koff_idx, koff in enumerate(koff_range):
        # simulate
        model.parameters[koff_name].value = koff
        y = dose_response(model, c_range, c_name, t_end, n_runs)
        mean_traj = np.mean(y[obs_name], axis=0)
        # std_traj = np.std(y[obs_name], axis=0)

        # find response
        c_typical_idx = np.argmin(np.abs(mean_traj - obs_thrs))
        if c_typical_idx == 0 or c_typical_idx == len(c_range):
            warnings.warn("c_range may not be large enough to find appropriate c needed to reach obs_thrs")
        c_typical[koff_idx] = c_range[c_typical_idx]

        # update progress to user
        print("{:.1f}%".format(100 * (koff_idx+1) / len(koff_range)))
    return np.array(c_typical)


if __name__ == '__main__':
    model_type = 'dimeric'
    plot_dr = True
    plot_rc = False
    crange = np.logspace(0, 4, 15) * 1E-12*1E-5*6.022E23
    koffrange = 1 / np.arange(3, 20, 2)
    t_end = 40
    num_traj = 50

    # --------------------------------------------------------------------------
    if model_type == 'adaptive_sorting':
        model = as_model
        n_threshold = 1.
    elif model_type == 'allosteric':
        model = allo_model
        n_threshold = 2E2
    elif model_type == 'dimeric':
        model = dimeric_model
        n_threshold = 2E2
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
        response = response_curve(model, crange, 'L_0', koffrange, 'koff', n_threshold, 'Cn', t_end, num_traj)
        fig, ax = plt.subplots()
        plt.plot(1 / koffrange, response, 'k--')
        plt.yscale('log')
        plt.xlabel(r'$\tau$ (s)')
        plt.ylabel('[Ligand]')
        plt.show()
