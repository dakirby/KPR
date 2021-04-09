import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
from copy import deepcopy
from pysb.simulator.bng import BngSimulator
import warnings

from allosteric import model as allo_model
from KPR1 import model as KPR1_model
from adaptive_sorting import model as as_model
from dimeric import model as dimeric_model
from trimeric import model as trimeric_model

# Mathematica-style plotting
FRAMECOLOR = '#737373'
matplotlib.rcParams['text.color'] = FRAMECOLOR
matplotlib.rcParams['axes.labelcolor'] = FRAMECOLOR
matplotlib.rcParams['xtick.color'] = FRAMECOLOR
matplotlib.rcParams['ytick.color'] = FRAMECOLOR
matplotlib.rcParams['axes.edgecolor'] = FRAMECOLOR

COLOURLIST = [(0.368417, 0.506779, 0.709798), (0.880722, 0.611041, 0.142051),
              (0.922526, 0.385626, 0.209179), (0.560181, 0.691569, 0.194885),
              (0.528488, 0.470624, 0.701351), (0.772079, 0.431554, 0.102387)]


def HN_ratio(model, delta, koff_name, obs_name, t_end, **kwargs):
    """Compute the ratio of levels of variable obs_name at time t_end for the
    same model with two different koff separated by factor delta.

    Return a sampling of this ratio for n_runs different stochastic runs of the
    model.
    """
    num_times = kwargs.get('num_times', 10)
    n_runs = kwargs.get('n_runs', 50)
    if type(koff_name) == list:
        multi_flag = True
        basline_koff = deepcopy([model.parameters[name].value for name in koff_name])

    else:
        multi_flag = False
        basline_koff = deepcopy(model.parameters[koff_name].value)
    t = np.linspace(0, t_end, num_times)

    # first koff
    sim1 = BngSimulator(model)
    x1 = sim1.run(tspan=t, verbose=False, n_runs=n_runs, method='ssa')
    y1 = np.array(x1.observables)[obs_name][:, -1]  # last time point for t.c.
    # second koff
    if multi_flag:
        for name in koff_name:
            model.parameters[name].value = model.parameters[name].value * delta
    else:
        model.parameters[koff_name].value = model.parameters[koff_name].value * delta

    sim2 = BngSimulator(model)
    x2 = sim2.run(tspan=t, verbose=False, n_runs=n_runs, method='ssa')
    y2 = np.array(x2.observables)[obs_name][:, -1]  # last time point for t.c.

    # reset model
    if multi_flag:
        for idx, name in enumerate(koff_name):
            model.parameters[name].value = basline_koff[idx]
    else:
        model.parameters[koff_name].value = basline_koff

    # discard any runs which ended with zero particles, to avoid infs
    del_list = []
    for idx in range(len(y1)):
        if y1[idx] == np.float64(0) or y2[idx] == np.float64(0):
            del_list.append(idx)
    if len(del_list) == len(y1):
        warnings.warn("Some choices of parameter produced no signal.")
    y1 = np.delete(y1, del_list)
    y2 = np.delete(y2, del_list)

    return y2 / y1


def HN_curve(model, params, **kwargs):
    """ model: one of the PySB models imported at the top of this file
        params: a dict with custom parameters for model
    """
    n_runs = kwargs.get('n_runs', 50)
    koff_name = kwargs.get('koff_name', 'koff')
    obs_name = kwargs.get('obs_name', 'Cn')
    t_end = kwargs.get('t_end', 1E3)

    # update model with params
    for key, val in params.items():
        if key in model.parameters.keys():
            model.parameters[key].value = val

    # compute curve
    mean_line = np.ones(len(delta_range))
    std_line = np.zeros(len(delta_range))
    for didx, d in enumerate(delta_range):
        ratio_sample = HN_ratio(model, d, koff_name, obs_name, t_end, n_runs=n_runs)
        mean_line[didx] = np.mean(ratio_sample)
        std_line[didx] = np.std(ratio_sample)

    # prepare curves for plotting
    upper = mean_line + std_line
    lower = mean_line - std_line
    for i in range(len(lower)):
        if lower[i] < 0:
            lower[i] = 0.1

    return mean_line, upper, lower


if __name__ == '__main__':
    # set up parameters
    MODELS = [allo_model, KPR1_model, dimeric_model]
    LABELS = ["Naive", r"KPR$_1$", "Dimeric"]

    # simulation parameters
    delta_range = np.logspace(-3, 0, num=15)
    n_runs = 1000

    # model parameters
    c = 1E-9
    koff = 1
    kf = 0.001
    kon = 2E5
    R0 = 1E2
    NA = 6.023E23
    tissue_cell_density = 1E5  # 1 billion cells per mL in tissue
    volEC = 1 / tissue_cell_density
    radius = 1E-5
    volPM = 4 * 3.14 * radius**2

    params = {'R_0': R0,
              'R1_0': R0/2, 'R2_0': R0/2,
              'L_0': int(c*NA*volEC),
              'kp': 1, 'ku': 0,
              'kappa': kon/(NA*volEC), 'koff': koff, 'kf': kf,
              'ka1': kon/(NA*volEC), 'ka2': kon/(NA*volEC),
              'kd1': koff, 'kd2': koff,
              'ka3': kf, 'ka4': kf,
              'kd3': koff, 'kd4': koff}

    # simulation
    lines = []
    for idx, model in enumerate(MODELS):
        print(LABELS[idx])
        if LABELS[idx] == "Dimeric":
            koff_name = ['kd1', 'kd2', 'kd3', 'kd4']
            mean_line, upper, lower = HN_curve(model, params, n_runs=n_runs, obs_name='Nobs', koff_name=koff_name)
        elif LABELS[idx] == r"KPR$_1$":
            mean_line, upper, lower = HN_curve(model, params, n_runs=n_runs, obs_name='Nobs', t_end=1E5)
        else:
            mean_line, upper, lower = HN_curve(model, params, n_runs=n_runs, obs_name='Nobs')
        lines.append((mean_line, upper, lower))

    # plotting
    fig, ax = plt.subplots(figsize=(7, 4))
    for i in range(len(lines)):
        plt.plot(delta_range, lines[i][0], color=COLOURLIST[i], label=LABELS[i])
        ax.fill_between(delta_range, lines[i][1], lines[i][2], color=COLOURLIST[i], alpha=0.1)
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(bottom=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlabel(r'Relative binding strength $\delta$')
    plt.ylabel(r'$\eta_{HN} = \frac{C_{N}[\delta \cdot k_{off}]}{C_{N}[k_{off}]}$')
    plt.title('Signal gain in strong proofreading regime')

    plt.legend(loc=6, bbox_to_anchor=(1, 0.5), frameon=False)
    plt.tight_layout()
    #plt.show()
    plt.savefig('output' + os.sep + 'Signal_gain_vs_delta.pdf')
