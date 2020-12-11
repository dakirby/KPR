import matplotlib.pyplot as plt
import numpy as np

from params import Params, DEFAULT_PARAMS
from settings import RECEPTOR_STATE_SIZE, INIT_CONDS
from trajectory_plotting import plot_traj_and_mean_sd, plot_means, plot_vars, plot_hist, plot_estimation
from trajectory_simulate import multitraj

plt.style.use('parameters.mplstyle')


def get_state_at_t(traj, times, t, last_step=0):
    for idx, i in enumerate(times):
        if i > t:
            return traj[idx-1], idx-1
        elif i == t:
            return traj[idx], idx
    print("The final time step for the simulation was {:.2f}".format(times[-1]))
    raise IndexError("The simulation did not reach the requested time point. Try running the simulation again with more time steps.")


def get_moment_timeseries(traj_array, times_array, params, model, moment_times=[]):
    """
    Returns, as a dict, multiple output timeseries (aligned to moment_times):
    return:  mean(n)(t), var(n)(t), distribution(n)(t), estimate_x(t)
    """
    # prepare moment times
    num_traj = np.shape(traj_array)[-1]
    dt = np.mean(times_array[1, :])
    endtime = np.min(times_array[-1, :])
    if moment_times == []:
        moment_times = np.arange(0.0, endtime, dt)
    # pass previous step to get_state_at_t(...) to speedup
    last_step = np.zeros(num_traj, dtype=int)
    # prepare value dict
    moment_curves = {'mean_n': None,
                     'var_n': None,
                     'distribution_n': None}

    n_idx = {'mode_1': 1, 'kpr': 2, 'adaptive_sorting': 3, 'dimeric': 5}[model]
    moment_curves['mean_n'] = np.zeros(len(moment_times))
    moment_curves['var_n'] = np.zeros(len(moment_times))
    moment_curves['distribution_n'] = np.zeros((len(moment_times), num_traj), dtype=int)

    for idx, t in enumerate(moment_times):
        statesum = 0.0
        statesquaresum = 0.0
        for k in range(num_traj):
            state_at_t, step = get_state_at_t(traj_array[:, :, k], times_array[:, k], t, last_step=last_step[k])
            last_step[k] = step
            statesum += state_at_t[n_idx]
            statesquaresum += state_at_t[n_idx]**2
            # store n(t) and m(t) for each trajectory to get histogram evolution
            moment_curves['distribution_n'][idx][k] = state_at_t[n_idx]

        moment_curves['mean_n'][idx] = statesum / num_traj
        moment_curves['var_n'][idx] = statesquaresum / num_traj - moment_curves['mean_n'][idx]**2
    return moment_curves, moment_times


if __name__ == '__main__':
    # settings
    model = 'mode_1'
    #model = 'kpr'
    #model = 'adaptive_sorting'

    for model in ['mode_1', 'kpr', 'adaptive_sorting']:
    #for model in [model]:
        print(model)

        num_traj = 500
        num_steps = 10000

        # model specification
        params = DEFAULT_PARAMS
        receptor_init_cond = np.zeros(RECEPTOR_STATE_SIZE[model])
        receptor_init_cond[0] = 1.0
        # simulate trajectories
        traj_array, times_array = multitraj(num_traj, num_steps=num_steps, model=model, params=params,
                                            bound_probabilities=receptor_init_cond)

        # compute moments from data
        simdata, moment_times = get_moment_timeseries(traj_array, times_array, params, model)

        # specify histogram timepoints
        num_points = 20
        hist_steps = [i*num_steps/num_points for i in range(num_points)]
        hist_steps = [-1]  # override the above line, just plot last time step

        # model dependent plotting
        for step in hist_steps:
            plot_hist(moment_times, simdata['distribution_n'], step, model, state_label='n', show=False)
