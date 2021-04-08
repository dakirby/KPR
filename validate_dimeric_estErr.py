import matplotlib.pyplot as plt
import numpy as np
import os
from pysb.simulator.bng import BngSimulator
import warnings
from differentiation_methods import spline_method

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


def cEst_dimeric(T, model):
    Rt = model.parameters['R1_0'].value + model.parameters['R2_0'].value
    Delta = model.parameters['R1_0'].value - model.parameters['R2_0'].value
    K1 = model.parameters['kd1'].value / model.parameters['ka1'].value
    K2 = model.parameters['kd2'].value / model.parameters['ka2'].value
    K4 = model.parameters['kd4'].value / model.parameters['ka4'].value
    numerator_t1 = K1*Rt**2 - 4*K1*K4*T - 4*K2*K4*T - 4*K1*Rt*T + 4*K1*T**2-K1*Delta**2
    numerator_t2 = -64*K1*K2*(K4**2)*(T**2) + (-K1*Rt**2 + 4*K1*K4*T + 4*K2*K4*T + 4*K1*Rt*T - 4*K1*T**2 + K1*Delta**2)**2
    return (numerator_t1 - np.sqrt(numerator_t2)) / (8*K4*T)


if __name__ == '__main__':
    model_type = 'dimeric'
    t_end = 500
    num_traj = 500

    # --------------------------------------------------------------------------
    if model_type == 'dimeric':
        model = dimeric_model
        crange = np.logspace(0, 3, 15) * 1E-9*1E-5*6.022E23
        est_fn = cEst_dimeric
    else:
        raise NotImplementedError

    print("Simulating")
    y = dose_response(model, crange, 'L_0', t_end, num_traj)
    print("Computing statistics")
    mean_traj = np.mean(y['Cn'], axis=0)
    var_traj = np.var(y['Cn'], axis=0)
    std_traj = np.sqrt(var_traj)

    print("Plotting")
    fig, ax = plt.subplots()
    plt.plot(crange, mean_traj)
    ax.fill_between(crange, mean_traj + std_traj, mean_traj - std_traj, 'b', alpha=0.2)
    # theory says variance in Cn should be Var(Cn) = Cn
    plt.plot(crange, mean_traj + np.sqrt(mean_traj), 'k--')
    plt.plot(crange, mean_traj - np.sqrt(mean_traj), 'k--')
    plt.xscale('log')
    plt.xlabel('c')
    plt.ylabel('Signal (T)')
    plt.title('Dimeric Receptor')
    plt.savefig('output'+os.sep+'dimeric_variance_validation.pdf')
