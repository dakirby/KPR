from pysb.simulator.bng import BngSimulator
import numpy as np
import warnings


def time_course(model, t_end, n_runs, num_times=10, params=None):
    assert n_runs > 1
    t = np.linspace(0, t_end, num_times)
    if params is not None:
        for key, val in params.items():
            model.parameters[key].value = val
    # simulate
    sim = BngSimulator(model)
    x = sim.run(tspan=t, verbose=False, n_runs=n_runs, method='ssa')
    y = np.array(x.observables)
    return y


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
