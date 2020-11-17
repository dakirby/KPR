import matplotlib.pyplot as plt
import numpy as np

from params import Params, DEFAULT_PARAMS
from settings import RECEPTOR_STATE_SIZE, INIT_CONDS
from trajectory_plotting import plot_traj_and_mean_sd, plot_means, plot_vars, plot_hist, plot_estimation
from trajectory_simulate import multitraj
from trajectory_analysis import get_moment_timeseries, get_state_at_t
from formulae import mode1_meanN_theory, kpr_meanN_theory

plt.style.use('parameters.mplstyle')
DEBUGGING = True


if __name__ == '__main__':
    # settings
    #model = 'mode_1'
    #model = 'kpr'
    model = 'adaptive_sorting'

    # FIM = (dn/dkoff)**2 / Var(n)
    taurange = np.arange(0.02, 1., 0.02)
    koffrange = [1./t for t in taurange]
    num_test_koff = len(koffrange)

    test_time = 1E2
    num_traj = 500
    num_steps = 5000

    # -------------------------------------------------------------------------
    # ---------- END USER CONTROL ---------------------------------------------
    # -------------------------------------------------------------------------
    # find mean and variance for each test affinity
    print('Testing range of koff')
    for idx, koff in enumerate(koffrange):
        # perform multi-trajectory simulation
        params = DEFAULT_PARAMS
        params.k_off = koff

        # model specification
        receptor_init_cond = np.zeros(RECEPTOR_STATE_SIZE[model])
        receptor_init_cond[0] = 1.0

        # simulate trajectories
        traj_array, times_array = multitraj(num_traj, num_steps=num_steps, model=model, params=params,
                                            bound_probabilities=receptor_init_cond)

        # compute moments from data
        simdata, moment_times = get_moment_timeseries(traj_array, times_array, params, model)
        meanN_sim = simdata['mean_n']
        varN_sim = simdata['var_n']

        mean_n_at_test_time, _ = get_state_at_t(meanN_sim, moment_times, test_time)

        # update progress to user
        percent_done = (idx + 1) / num_test_koff * 100
        print("{:.2f}".format(percent_done) + '% done')

        # for debugging purposes
        if DEBUGGING == True:
            if model == 'mode_1':
                print("<n>(kp*t={}) = {}".format(params.k_p * test_time, mean_n_at_test_time))

                kpt_axis = moment_times * params.k_p
                estimate_n = mode1_meanN_theory(params, moment_times)
                plt.plot(kpt_axis, meanN_sim)
                plt.plot(kpt_axis, estimate_n)
                plt.xlabel('kp*t')
                plt.ylabel('n')
                plt.title('Naive receptor')
                plt.show()
            elif model == 'kpr':
                print("<n>(kp*t={}) = {}".format(params.k_p * test_time, mean_n_at_test_time))

                kpt_axis = moment_times * params.k_p
                estimate_n = kpr_meanN_theory(params, moment_times)
                plt.plot(kpt_axis, meanN_sim)
                plt.plot(kpt_axis, estimate_n)
                plt.xlabel('kp*t')
                plt.ylabel('n')
                plt.title('KPR receptor')
                plt.show()

            elif model == 'adaptive_sorting':
                print("<n>(kp*t={}) = {}".format(params.k_p * test_time, mean_n_at_test_time))

                kpt_axis = moment_times * params.k_p
                estimate_n = adaptive_sorting_meanN_theory(params, moment_times)
                plt.plot(kpt_axis, meanN_sim)
                plt.plot(kpt_axis, estimate_n)
                plt.xlabel('kp*t')
                plt.ylabel('n')
                plt.title('Adaptive Sorting')
                plt.show()

            exit()
