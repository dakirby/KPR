import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from params import Params, DEFAULT_PARAMS
from settings import RECEPTOR_STATE_SIZE, INIT_CONDS, FOLDER_OUTPUT, RECEPTOR_BOUND_PROBABILITIES
from trajectory_plotting import plot_traj_and_mean_sd, plot_means, plot_vars, plot_hist, plot_estimation
from trajectory_simulate import multitraj
from trajectory_analysis import get_moment_timeseries, get_state_at_t
from formulae import *

plt.style.use('parameters.mplstyle')

if __name__ == '__main__':
    DEBUGGING = False

    # settings
    model = 'mode_1'
    #model = 'kpr'
    #model = 'adaptive_sorting'

    # FIM = (dn/dkoff)**2 / Var(n)
    dKOFF = 0.1
    koffrange = np.arange(1E1, 3E1, dKOFF)
    num_test_koff = len(koffrange)

    test_time = 1E1
    # The number of trajectories and steps needed for each model for the above
    # choice for test_time
    SIM_PARAMS = {'mode_1': [int(1E3), 200],
                  'kpr': [int(1E5), 200],
                  'adaptive_sorting': []}

    num_traj = SIM_PARAMS[model][0]
    num_steps = SIM_PARAMS[model][1]

    # plot a tangent line 20% of the way into koffrange
    tangent_idx = int(0.2 * num_test_koff)

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

        # simulate trajectories
        traj_array, times_array = multitraj(num_traj, num_steps=num_steps, model=model, params=params,
                                            bound_probabilities=RECEPTOR_BOUND_PROBABILITIES[model])

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
                # Mean n
                print("<n>(kp*t={}) = {}".format(params.k_p * test_time, mean_n_at_test_time))

                kpt_axis = moment_times * params.k_p
                estimate_n = mode1_meanN_theory(params, moment_times)
                plt.plot(kpt_axis, meanN_sim, label='Simulation')
                plt.plot(kpt_axis, estimate_n, label='Theory')
                plt.xlabel('kp*t')
                plt.ylabel(r'$\langle n \rangle$')
                plt.title('Naive receptor')
                plt.legend()
                plt.show()

                # Variance in n
                print("Var(n)(kp*t={}) = {}".format(params.k_p * test_time, var_n_at_test_time))

                kpt_axis = moment_times * params.k_p
                estimate_n_var = mode1_varN_theory(params, moment_times)
                plt.plot(kpt_axis, varN_sim, label='Simulation')
                plt.plot(kpt_axis, estimate_n_var, label='Theory')
                plt.xlabel('kp*t')
                plt.ylabel('Var(n)')
                plt.title('Naive receptor')
                plt.legend()
                plt.show()

            elif model == 'kpr':
                # Mean n
                print("<n>(kp*t={}) = {}".format(params.k_p * test_time, mean_n_at_test_time))

                kpt_axis = moment_times * params.k_p
                estimate_n = kpr_meanN_theory(params, moment_times)
                plt.plot(kpt_axis, meanN_sim, label='Simulation')
                plt.plot(kpt_axis, estimate_n, label='Theory')
                plt.xlabel('kp*t')
                plt.ylabel('n')
                plt.title('KPR receptor')
                plt.legend()
                plt.show()

                # Variance in n
                print("Var(n)(kp*t={}) = {}".format(params.k_p * test_time, var_n_at_test_time))

                kpt_axis = moment_times * params.k_p
                estimate_n_var = kpr_varN_theory(params, moment_times)
                plt.plot(kpt_axis, varN_sim, label='Simulation')
                plt.plot(kpt_axis, estimate_n_var, label='Theory')
                plt.xlabel('kp*t')
                plt.ylabel('Var(n)')
                plt.title('KPR receptor')
                plt.legend()
                plt.show()


            elif model == 'adaptive_sorting':
                print("<n>(kp*t={}) = {}".format(params.k_p * test_time, mean_n_at_test_time))

                kpt_axis = moment_times * params.k_p
                estimate_n = adaptive_sorting_meanN_theory(params, moment_times)
                plt.plot(kpt_axis, meanN_sim, label='Simulation')
                plt.plot(kpt_axis, estimate_n, label='Theory')
                plt.xlabel('kp*t')
                plt.ylabel('n')
                plt.title('Adaptive Sorting')
                plt.legend()
                plt.show()

            exit()

        # Write record to file
        df = pd.DataFrame(record, columns=["koff", "mean_N", "var_N"])
        df.to_csv(FOLDER_OUTPUT + os.sep + '{}_n_data.csv'.format(model))

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

                delta_sq = record[idx][2] / (grad**2)
                delta_koff_sq.append([record[idx][0], delta_sq])

        df2 = pd.DataFrame(delta_koff_sq, columns=["koff", "delta_koff_sq"])
        df2.to_csv(FOLDER_OUTPUT + os.sep + '{}_delta_koff_sq.csv'.format(model))

    # Compare to theory
    fig, axes = plt.subplots(nrows=2, ncols=2)
    if model in ["mode_1", "kpr"]:
        koff_axis = [el[0] for el in record]
        theory_n = []
        theory_var = []
        for koff in koff_axis:
            params.k_off = koff
            theory_n.append(mode1_meanN_theory(params, [test_time])[0])
            theory_var.append(mode1_varN_theory(params, [test_time])[0])
        # Mean n
        axes[0, 0].plot(koff_axis, [el[1] for el in record], label='Simulation')
        axes[0, 0].plot(koff_axis, theory_n, label='Theory')
        axes[0, 0].set_xlabel(r'$k_{off}$')
        axes[0, 0].set_ylabel(r'$\langle n \rangle$')
        axes[0, 0].set_title('Mean')

        # Variance in n
        axes[0, 1].plot(koff_axis, [el[2] for el in record], label='Simulation')
        axes[0, 1].plot(koff_axis, theory_var, label='Theory')
        axes[0, 1].set_xlabel(r'$k_{off}$')
        axes[0, 1].set_ylabel(r'Var($n$)')
        axes[0, 1].set_title('Variance')
        axes[0, 1].legend()


        # Gradient
        delta_koff_sq = np.array(delta_koff_sq)
        axes[1, 0].scatter(delta_koff_sq[:, 0], delta_koff_sq[:, 1] / np.square(delta_koff_sq[:, 0]), label='simulation')

        min_koff = min(delta_koff_sq[:, 0])
        max_koff = max(delta_koff_sq[:, 0])
        theory_line = []
        for idx, koff in enumerate(np.logspace(np.log10(min_koff), np.log10(max_koff), 50)):
            params = DEFAULT_PARAMS
            params.k_off = koff
            theory_line.append([koff, mode1_relKOFF_theory(params, [test_time])[0]])
        theory_line = np.array(theory_line)
        axes[1, 0].plot(theory_line[:, 0], theory_line[:, 1], 'g', label='theory')
        #axes[1, 0].set_xscale('log')
        axes[1, 0].set_xlim([min(koff_axis), max(koff_axis)])
        axes[1, 0].legend()
        axes[1, 0].set_xlabel(r'$k_{off}$')
        axes[1, 0].set_ylabel(r'$\delta k_{off}^2 / k_{off}^2$')
        axes[1, 0].set_title('Estimation Error')

        # Add tangent line to mean n plot
        tangent = []
        grad = dmu_dkoff[tangent_idx][1]
        mu_idx = np.where(np.array(record)[:, 0] == dmu_dkoff[tangent_idx][0])[0][0]
        intercept_y = record[mu_idx][1]
        intercept_x = record[mu_idx][0]

        num_tangent_pts = max(3, int(0.1 * num_test_koff))
        for offset in range(int(-num_tangent_pts / 2), int(num_tangent_pts / 2) + 1):
            pt = intercept_y + grad * (record[mu_idx + offset][0] - intercept_x)
            tangent.append([record[mu_idx + offset][0], pt])
        tangent = np.array(tangent)

        axes[0, 0].plot(tangent[:, 0], tangent[:, 1], '--', label='numerical tangent')
        axes[0, 0].legend()
        plt.savefig(FOLDER_OUTPUT + os.sep + '{}_numdiff_validation.pdf'.format(model))
        plt.show()
