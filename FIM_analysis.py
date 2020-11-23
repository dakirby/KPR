import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from params import Params, DEFAULT_PARAMS
from settings import RECEPTOR_STATE_SIZE, INIT_CONDS, FOLDER_OUTPUT
from trajectory_plotting import plot_traj_and_mean_sd, plot_means, plot_vars, plot_hist, plot_estimation
from trajectory_simulate import multitraj
from trajectory_analysis import get_moment_timeseries, get_state_at_t
from formulae import mode1_meanN_theory, kpr_meanN_theory, adaptive_sorting_meanN_theory, mode1_relKOFF_theory

plt.style.use('parameters.mplstyle')


if __name__ == '__main__':
    DEBUGGING = False

    # settings
    model = 'mode_1'
    #model = 'kpr'
    #model = 'adaptive_sorting'

    # FIM = (dn/dkoff)**2 / Var(n)
    dKOFF = 0.5
    koffrange = np.arange(1E-2, 1E2, dKOFF)
    taurange = [1./koff for koff in koffrange]
    num_test_koff = len(koffrange)

    test_time = 1E1
    num_traj = 500
    num_steps = 200

    # -------------------------------------------------------------------------
    # ---------- END USER CONTROL ---------------------------------------------
    # -------------------------------------------------------------------------
    # find mean and variance for each test affinity
    print('Testing range of koff')
    record = []
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
        if DEBUGGING: print("Done simulations");

        # compute moments from data
        simdata, moment_times = get_moment_timeseries(traj_array, times_array, params, model)
        if DEBUGGING: print("Done collecting moment timeseries");

        meanN_sim = simdata['mean_n']
        varN_sim = simdata['var_n']

        mean_n_at_test_time, _ = get_state_at_t(meanN_sim, moment_times, test_time, last_step=len(moment_times)-10)
        var_n_at_test_time, _ = get_state_at_t(varN_sim, moment_times, test_time, last_step=len(moment_times)-10)

        record.append([koff, mean_n_at_test_time, var_n_at_test_time])

        # update progress to user
        percent_done = (idx + 1) / num_test_koff * 100
        print("{:.2f}".format(percent_done) + '% done')

        # for debugging purposes
        if DEBUGGING:
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

        # Write record to file
        df = pd.DataFrame(record, columns=["koff", "mean_N", "var_N"])
        df.to_csv(FOLDER_OUTPUT + os.sep + '{}_n_data'.format(model))

        # Derivative analysis
        dmu_dkoff = []
        delta_koff_sq = []
        for idx in range(len(record)):
            if idx == 0 or idx == len(record) - 1:  # can't use secant method on first or last point
                pass
            else:
                mu_1 = record[idx + 1][1]
                mu_2 = record[idx - 1][1]
                grad = (mu_1 - mu_2) / (2 * dKOFF)
                dmu_dkoff.append([record[idx][0], grad])

                delta_sq = record[idx][2] / grad**2
                delta_koff_sq.append([record[idx][0], delta_sq])

        df2 = pd.DataFrame(delta_koff_sq, columns=["koff", "delta_koff_sq"])
        df2.to_csv(FOLDER_OUTPUT + os.sep + '{}_delta_koff_sq'.format(model))

    # Compare to theory
    if model in ["mode_1", "kpr"]:
        delta_koff_sq = np.array(delta_koff_sq)
        plt.scatter(delta_koff_sq[:, 0], delta_koff_sq[:, 1] / np.square(delta_koff_sq[:, 0]), label='simulation')

        min_koff = min(delta_koff_sq[:, 0])
        max_koff = max(delta_koff_sq[:, 0])
        theory_line = []
        for idx, koff in enumerate(np.logspace(np.log10(min_koff), np.log10(max_koff), 50)):
            params = DEFAULT_PARAMS
            params.k_off = koff
            theory_line.append([koff, mode1_relKOFF_theory(params, [test_time])[0]])
        theory_line = np.array(theory_line)
        plt.plot(theory_line[:, 0], theory_line[:, 1], 'g', label='theory')
        plt.xscale('log')
        plt.legend()
        plt.xlabel(r'$k_{off}$')
        plt.ylabel(r'$\delta k_{off}^2 / k_{off}^2$')
        plt.show()
