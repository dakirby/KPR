import matplotlib.pyplot as plt
import numpy as np
import os
from copy import deepcopy
from pysb.simulator.bng import BngSimulator

from allosteric import model as allo_model
from KPR1 import model as KPR1_model
from adaptive_sorting import model as as_model
from dimeric import model as dimeric_model
from trimeric import model as trimeric_model


def HN_ratio(model, delta, koff_name, obs_name, t_end, **kwargs):
    basline_koff = deepcopy(model.parameters[koff_name].value)
    num_times = kwargs.get('num_times', 10)
    n_runs = kwargs.get('n_runs', 50)
    t = np.linspace(0, t_end, num_times)

    # first koff
    sim1 = BngSimulator(model)
    x1 = sim1.run(tspan=t, verbose=False, n_runs=n_runs, method='ssa')
    y1 = np.array(x1.observables)[obs_name][:, -1]  # last time point for t.c.
    # second koff
    model.parameters[koff_name].value = model.parameters[koff_name].value * delta
    sim2 = BngSimulator(model)
    x2 = sim2.run(tspan=t, verbose=False, n_runs=n_runs, method='ssa')
    y2 = np.array(x2.observables)[obs_name][:, -1]  # last time point for t.c.

    model.parameters[koff_name].value = basline_koff  # reset model
    return y2 / y1


def HN_curve(model, params, **kwargs):
    """ model: one of the PySB models imported at the top of this file
        params: a dict with custom parameters for model
    """
    n_runs = kwargs.get('n_runs', 50)
    for key, val in params.items():
        if key in model.parameters.keys():
            model.parameters[key].value = val

    mean_line = np.ones(len(delta_range))
    std_line = np.zeros(len(delta_range))
    for didx, d in enumerate(delta_range):
        ratio_sample = HN_ratio(model, d, 'koff', 'Cn', 10, n_runs=n_runs)
        mean_line[didx] = np.mean(ratio_sample)
        std_line[didx] = np.std(ratio_sample)

    # plotting
    upper = mean_line + std_line
    lower = mean_line - std_line
    for i in range(len(lower)):
        if lower[i] < 0:
            lower[i] = 0.1

    return mean_line, upper, lower


COLOURLIST = [(0.368417, 0.506779, 0.709798), (0.880722, 0.611041, 0.142051),
              (0.560181, 0.691569, 0.194885), (0.922526, 0.385626, 0.209179),
              (0.528488, 0.470624, 0.701351), (0.772079, 0.431554, 0.102387)]


if __name__ == '__main__':
    # set up parameters
    delta_range = np.logspace(-3, 0, num=15)
    c = 1E-7
    NA = 6.023E23
    tissue_cell_density = 1E9 / 1E-3  # 1 billion cells per mL
    volEC = 1 / tissue_cell_density
    params = {'R_0': 1E2, 'L_0': int(c*NA*volEC), 'kappa': 1E6/(NA*volEC), 'koff': 1, 'kf': 1E-3}

    # simulation
    lines = []
    for idx, model in enumerate([allo_model, KPR1_model]):
        mean_line, upper, lower = HN_curve(model, params)
        print(mean_line)
        lines.append((mean_line, upper, lower))

    # plotting
    fig, ax = plt.subplots()
    for i in range(len(lines)):
        plt.plot(delta_range, lines[i][0], color=COLOURLIST[i])
        ax.fill_between(delta_range, lines[i][1], lines[i][2], color=COLOURLIST[i], alpha=0.1)
    plt.xscale('log')
    plt.yscale('log')
    plt.show()
