import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from multiprocessing import cpu_count
from tqdm.contrib.concurrent import process_map

from params import DEFAULT_PARAMS
from settings import RECEPTOR_STATE_SIZE, FOLDER_OUTPUT, RECEPTOR_BOUND_PROBABILITIES, GLOB_KA1
from trajectory_simulate import multitraj
from trajectory_analysis import get_moment_timeseries, get_state_at_t
from formulae import *
from differentiation_methods import polynomial_method, spline_method, secant_method, inverse_polynomial_method

plt.style.use('parameters.mplstyle')

# -------------------------------------------------------------------------
# ---------- END USER CONTROL ---------------------------------------------
# -------------------------------------------------------------------------
DEBUGGING = False

# Model to investigate
#model = 'mode_1'
#model = 'kpr'
#model = 'adaptive_sorting'
model = 'dimeric'

# Derivative method
#method = 'secant'
#method = 'poly'
method = 'spline'
#method = 'invpoly'

METHODS = {'secant': secant_method,
           'spline': spline_method,
           'poly': polynomial_method,
           'invpoly': inverse_polynomial_method}

# K_off sampling scheme
if model == 'dimeric':
    dKOFF = 0.0025 * GLOB_KA1
    koffrange = np.arange(0.001 * GLOB_KA1, 5*GLOB_KA1, dKOFF)
else:
    dKOFF = 3
    koffrange = np.arange(1E1, 5E1, dKOFF)
num_test_koff = len(koffrange)

TEST_TIME = 1E1
# Simulation paramters
SIM_PARAMS = {'mode_1': [int(1E3), 200],
              'kpr': [int(1E5), 200],
              'adaptive_sorting': [int(5E0), int(3E5)],
              'dimeric': [int(1E2), int(5E5)]}

num_traj = SIM_PARAMS[model][0]
num_steps = SIM_PARAMS[model][1]

# plot a tangent line 20% of the way into koffrange
tangent_idx = int(0.2 * num_test_koff)

# Multiprocessing
num_threads = cpu_count()

# -------------------------------------------------------------------------
# ---------- END USER CONTROL ---------------------------------------------
# -------------------------------------------------------------------------
def find_num_steps(num_traj, num_steps, model, RECEPTOR_BOUND_PROBABILITIES, TEST_TIME):
    """ Runs the simulation and plots the time course of the mean and variance,
    as a way to check that the simulation is being run for long enough and that
    the stochasticity is minimized by the number of trajectories being used.
    """
    params = DEFAULT_PARAMS

    # simulate trajectories
    traj_array, times_array = multitraj(num_traj, num_steps=num_steps, model=model, params=params,
                                        bound_probabilities=RECEPTOR_BOUND_PROBABILITIES[model])
    print("Done simulations")

    if np.min(times_array[-1, :]) < TEST_TIME:
        print("The final time step for the simulation was {:.2f}".format(np.min(times_array[-1, :])))
        raise IndexError("The simulation did not reach the requested time point. Try running the simulation again with more time steps.")
    else:
        print("The simulation appears to have been run for a sufficient number of steps.")

    # compute moments from data
    if model != 'adaptive_sorting':
        print("Collecting moment timeseries")
        simdata, moment_times = get_moment_timeseries(traj_array, times_array, params, model)
        print("Done collecting moment timeseries")
    else:  # speed-up for adaptive sorting because it is expensive to compute
        dt = np.mean(times_array[1, :])
        moment_times_input = np.array([TEST_TIME - dt, TEST_TIME, TEST_TIME + dt])
        simdata, moment_times = get_moment_timeseries(traj_array, times_array, params, model, moment_times=moment_times_input)

    meanN_sim = simdata['mean_n']
    varN_sim = simdata['var_n']
    mean_n_at_test_time, _ = get_state_at_t(meanN_sim, moment_times, TEST_TIME, last_step=len(moment_times)-10)
    var_n_at_test_time, _ = get_state_at_t(varN_sim, moment_times, TEST_TIME, last_step=len(moment_times)-10)

    if model == 'mode_1':
        # Mean n
        print("<n>(kp*t={}) = {}".format(params.k_p * TEST_TIME, mean_n_at_test_time))

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
        print("Var(n)(kp*t={}) = {}".format(params.k_p * TEST_TIME, var_n_at_test_time))

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
        print("<n>(kp*t={}) = {}".format(params.k_p * TEST_TIME, mean_n_at_test_time))

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
        print("Var(n)(kp*t={}) = {}".format(params.k_p * TEST_TIME, var_n_at_test_time))

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
        print("<n>(kp*t={}) = {}".format(params.k_p * TEST_TIME, mean_n_at_test_time))

        kpt_axis = moment_times * params.k_p
        estimate_n = adaptive_sorting_meanN_theory(params, moment_times)
        plt.plot(kpt_axis, meanN_sim, label='Simulation')
        #plt.plot(kpt_axis, estimate_n, label='Theory')
        plt.xlabel('kp*t')
        plt.ylabel('n')
        plt.title('Adaptive Sorting')
        plt.legend()
        plt.show()


def thread_func(koff):
    """ Finds the mean and variance at TEST_TIME for input koff. Other settings
    such as which model to use are defined globally, which is required becuase
    the function must be defined outside main() so that multiprocessing can be
    implemented.
    """
    params = DEFAULT_PARAMS
    params.k_off = koff

    # simulate trajectories
    traj_array, times_array = multitraj(num_traj, num_steps=num_steps, model=model, params=params,
                                        bound_probabilities=RECEPTOR_BOUND_PROBABILITIES[model])

    # compute moments from data
    dt = np.mean(times_array[1, :])
    moment_times_input = np.array([TEST_TIME - dt, TEST_TIME, TEST_TIME + dt])
    simdata, moment_times = get_moment_timeseries(traj_array, times_array, params, model, moment_times=moment_times_input)

    meanN_sim = simdata['mean_n']
    varN_sim = simdata['var_n']

    mean_n_at_test_time, _ = get_state_at_t(meanN_sim, moment_times, TEST_TIME, last_step=len(moment_times)-10)
    var_n_at_test_time, _ = get_state_at_t(varN_sim, moment_times, TEST_TIME, last_step=len(moment_times)-10)

    return [koff, mean_n_at_test_time, var_n_at_test_time]


if __name__ == '__main__':
    print(model)
    # find mean and variance for each test affinity
    print('Testing range of koff')
    if DEBUGGING:
        find_num_steps(num_traj, num_steps, model, RECEPTOR_BOUND_PROBABILITIES, TEST_TIME)

    else:
        # Multiprocessing to find moments as a function of koff
        record = process_map(thread_func, koffrange, max_workers=num_threads, chunksize=1)

        # Write record to file
        df = pd.DataFrame(record, columns=["koff", "mean_N", "var_N"])
        df.to_csv(FOLDER_OUTPUT + os.sep + '{}_n_data.csv'.format(model))

        # Derivative analysis
        print("Computing derivatives")
        dmu_dkoff, delta_koff_sq, mean_fit = METHODS[method](record)

        df2 = pd.DataFrame(delta_koff_sq, columns=["koff", "delta_koff_sq"])
        df2.to_csv(FOLDER_OUTPUT + os.sep + '{}_delta_koff_sq.csv'.format(model))

        if method == 'spline':
            print("Coefficients:")
            print(mean_fit.get_coeffs())
            print("Knots:")
            print(mean_fit.get_knots())
        # Compare to theory
        theory_dict = {"mode_1": [mode1_meanN_theory, mode1_varN_theory],
                       "kpr": [kpr_meanN_theory, kpr_varN_theory],
                       "adaptive_sorting": [adaptive_sorting_meanN_theory, None],
                       "dimeric": [dimeric_meanN_theory, None]}

        print("Plotting")
        fig, axes = plt.subplots(nrows=2, ncols=2)

        # Plot mean n
        koff_axis = [el[0] for el in record]
        axes[0, 0].plot(koff_axis, [el[1] for el in record], label='Simulation')

        if method in ['spline', 'poly', 'invpoly']:
            koff_fine_axis = np.arange(koff_axis[0], koff_axis[-1], dKOFF*0.1)
            fit = mean_fit(koff_fine_axis)
            axes[0, 0].plot(koff_fine_axis, fit, label=method)

        theory_n = []
        theory_var = []
        params = DEFAULT_PARAMS
        for koff in koff_axis:
            params.k_off = koff
            theory_n.append(theory_dict[model][0](params, [TEST_TIME])[0])
            if model in ["mode_1", "kpr"]:
                theory_var.append(theory_dict[model][1](params, [TEST_TIME])[0])
        axes[0, 0].plot(koff_axis, theory_n, label='Theory')

        axes[0, 0].set_xlabel(r'$k_{off}$')
        axes[0, 0].set_ylabel(r'$\langle n \rangle$')
        axes[0, 0].set_title('Mean')

        # Plot variance in n
        axes[0, 1].plot(koff_axis, [el[2] for el in record], label='Simulation')
        if model in ["mode_1", "kpr"]:
            axes[0, 1].plot(koff_axis, theory_var, label='Theory')
        axes[0, 1].set_xlabel(r'$k_{off}$')
        axes[0, 1].set_ylabel(r'Var($n$)')
        axes[0, 1].set_title('Variance')
        axes[0, 1].legend()

        # Plot relative error
        delta_koff_sq = np.array(delta_koff_sq)
        rel_koff = delta_koff_sq[:, 1] / np.square(delta_koff_sq[:, 0])
        axes[1, 1].scatter(delta_koff_sq[:, 0], rel_koff, label='simulation')
        print("Range of delta_koff is ({}, {})".format(min(rel_koff), max(rel_koff)))
        min_koff = min(delta_koff_sq[:, 0])
        max_koff = max(delta_koff_sq[:, 0])
        if model in ["mode_1", "kpr"]:
            rel_theory = {'mode_1': mode1_relKOFF_theory,
                          'kpr': kpr_relKOFF_theory}
            theory_line = []
            for idx, koff in enumerate(np.logspace(np.log10(min_koff), np.log10(max_koff), 50)):
                params = DEFAULT_PARAMS
                params.k_off = koff
                theory_line.append([koff, rel_theory[model](params, [TEST_TIME])[0]])
            theory_line = np.array(theory_line)
            axes[1, 1].plot(theory_line[:, 0], theory_line[:, 1], 'g', label='theory')

        axes[1, 1].set_xlim([min(koff_axis), max(koff_axis)])
        axes[1, 1].set_ylim(np.percentile(rel_koff, [0, 95]))  # contain 95% of data
        if max(rel_koff)/min(rel_koff) > 1000:
            axes[1, 1].set_yscale('log')
        axes[1, 1].legend()
        axes[1, 1].set_xlabel(r'$k_{off}$')
        axes[1, 1].set_ylabel(r'$\delta k_{off}^2 / k_{off}^2$')
        axes[1, 1].set_title('Estimation Error')

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

        # Plot dmu_dkoff
        axes[1, 0].scatter(list(zip(*dmu_dkoff))[0], list(zip(*dmu_dkoff))[1], label='simulation')
        theory_grad = {'mode_1': mode1_dNdKOFF_theory,
                       'kpr': kpr_dNdKOFF_theory,
                       'adaptive_sorting': adaptive_sorting_dNdKOFF_theory,
                       'dimeric': dimeric_dNdKOFF_theory}
        dmu_dkoff_theory_line = []
        for idx, koff in enumerate(np.logspace(np.log10(min_koff), np.log10(max_koff), 50)):
            params = DEFAULT_PARAMS
            params.k_off = koff
            dmu_dkoff_theory_line.append([koff, theory_grad[model](params, [TEST_TIME])[0]])
        dmu_dkoff_theory_line = np.array(dmu_dkoff_theory_line)
        axes[1, 0].plot(dmu_dkoff_theory_line[:, 0], dmu_dkoff_theory_line[:, 1], 'g', label='theory')
        axes[1, 0].set_xlabel(r'$k_{off}$')
        axes[1, 0].set_ylabel(r'$\partial \mu / \partial k_{off}$')
        axes[1, 0].set_title('Numerical Gradient')
        axes[1, 0].legend()

        plt.savefig(FOLDER_OUTPUT + os.sep + '{}_numdiff_validation.pdf'.format(model))
        plt.show()
