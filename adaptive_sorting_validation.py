import matplotlib.pyplot as plt
import numpy as np

from params import Params, DEFAULT_PARAMS
from settings import RECEPTOR_STATE_SIZE, INIT_CONDS, DEFAULT_MODEL, NUM_STEPS
from trajectory_plotting import plot_traj_and_mean_sd, plot_means, plot_vars, plot_hist, plot_estimation
from trajectory_simulate import multitraj
from trajectory_analysis import get_moment_timeseries, moment_dose_response

plt.style.use('parameters.mplstyle')


def response_curve(koff_range, c_range, num_traj, moment_times,
                   num_steps=NUM_STEPS, init_cond=None, model=DEFAULT_MODEL,
                   params=DEFAULT_PARAMS):
    """
    Returns, as a dict, multiple output timeseries (aligned to the threshold n
    determing response, then koff in koff_range, then concentrations in
    c_range, then moment_times):
    return:  response_curves (list) indexed as [n_threshold_idx, koff_idx, t_idx]
             n_curves (list)
    """
    min_n = 0.
    max_n = np.inf
    moment_curves_list = {'mean_n': [], 'var_n': [], 'distribution_n': []}
    # run simulations
    for koff in koff_range:
        dr_tc = moment_dose_response(c_range, num_traj, moment_times,
                                     init_cond, num_steps=num_steps,
                                     model=model, params=params)
        min_n_part = np.min(dr_tc['mean_n'])
        max_n_part = np.max(dr_tc['mean_n'])
        if min_n_part > min_n:
            min_n = min_n_part
        if max_n_part < max_n:
            max_n = max_n_part
        for key in moment_curves_list.keys():
            moment_curves_list[key].append(dr_tc[key])
    # convert to ndarray
    for key in moment_curves_list.keys():
        moment_curves_list[key] = np.array(moment_curves_list[key])
        # sample indexing: dict['mean_n'][koff_idx, c_idx, t_idx]
    # find response curves
    n_curves = list(range(int(min_n), int(max_n)))
    n_lookup = [] # indexed as [n_thrshld_idx, koff_idx, t_idx]
    for n_thrshld in n_curves:
        response_curves = np.zeros((len(koff_range), len(moment_times))) # indexed as [koff_idx, t_idx]
        for koff_idx, koff in enumerate(koff_range):
            for t_idx, t in enumerate(moment_times):
                diff = moment_curves_list['mean_n'][koff_idx, :, t_idx] - n_thrshld
                c_idx = np.argmin(np.abs(diff))
                L = moment_curves_list['mean_n'][koff_idx, c_idx, t_idx]
                response_curves[koff_idx, t_idx] = L
        n_lookup.append(response_curves)
    n_lookup = np.array(n_lookup)
    return n_lookup, np.array(n_curves)


if __name__ == '__main__':
    # settings
    model = 'mode_1'

    num_traj = 250
    num_steps = 1400
    koff_range = np.logspace(-9, 3, num=15)
    c_range = np.logspace(3, 8)
    moment_times = list(range(7))

    # model specification
    params = DEFAULT_PARAMS
    receptor_init_cond = np.zeros(RECEPTOR_STATE_SIZE[model])
    receptor_init_cond[0] = 1.0
    # simulate trajectories
    traj_array_list = []
    times_array = []

    # dose_response_tc = moment_dose_response(c_range, num_traj, moment_times,
    #                                         receptor_init_cond,
    #                                         num_steps=num_steps, model=model,
    #                                         params=params)
    # y1 = dose_response_tc['mean_n'][:, -1] + np.sqrt(dose_response_tc['var_n'][:, -1])
    # y2 = dose_response_tc['mean_n'][:, -1] - np.sqrt(dose_response_tc['var_n'][:, -1])
    # fig, ax = plt.subplots()
    # plt.plot(c_range, dose_response_tc['mean_n'][:, -1], 'k')
    # ax.fill_between(c_range, y1, y2, alpha=0.5)
    # plt.xscale('log')
    # plt.show()
    # exit()
    n_lookup, n_curves = response_curve(koff_range, c_range, num_traj,
                                        moment_times, num_steps=num_steps,
                                        init_cond=receptor_init_cond,
                                        model=model, params=params)
    mean_n_thrshld = int(np.mean(n_curves))
    thrshld_idx = np.argmin(np.abs(n_curves-mean_n_thrshld))
    plt.plot(1/koff_range, n_lookup[thrshld_idx, :, -1])
    plt.xlabel('tau')
    plt.ylabel('L')
    plt.xscale('log')
    plt.yscale('log')
    plt.show()
