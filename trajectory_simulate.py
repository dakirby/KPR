import matplotlib.pyplot as plt
import numpy as np

from params import DEFAULT_PARAMS
from settings import DEFAULT_MODEL, VALID_MODELS, NUM_RXN, NUM_STEPS, INIT_CONDS, STATE_SIZE, RECEPTOR_STATE_SIZE


def update_state(state_timeseries, rxn_idx, step, model):
    """
    Update the state based on the increment associated with the rxn_idx of specified model
    """
    current_state = state_timeseries[step]
    increment = UPDATE_DICTS[model][rxn_idx]
    state_timeseries[step + 1, :] = current_state + increment
    return state_timeseries


def propensities(state, model, params=DEFAULT_PARAMS):
    """
    Computes propensitiers of each rxn based on current state.
    Reaction labelling follows order defined in UPDATE_DICTS.
    """
    p = params
    assert model in VALID_MODELS
    propensities = np.zeros(NUM_RXN[model])
    if model == 'mode_1':
        propensities[0] = p.k_on * p.c * (1 - state[0])  # bind (0.0 if already bound)
        propensities[1] = p.k_off * state[0]             # unbind (0.0 if already unbound)
        propensities[2] = p.k_p * state[0]               # produce one n molecule
    elif model == 'kpr':
        propensities[0] = p.k_on * p.c * (1 - state[0]) * (1 - state[1])  # binding
        propensities[1] = p.k_off * state[0]                              # unbinding
        propensities[2] = p.k_f * state[0]                                # kpr forward step
        propensities[3] = p.k_off * state[1]                              # fall off
        propensities[4] = p.k_p * state[1]                                # produce n
    elif model == 'adaptive_sorting':
        # state = [0,    1,   2,   n,   K]
        propensities[0] = p.k_on * p.c * state[0]                         # binding
        propensities[1] = p.k_off * state[1]                              # unbinding
        propensities[2] = p.alpha * state[4] * state[1]                   # kpr sorting step
        propensities[3] = p.k_off * state[2]                              # fall off
        propensities[4] = p.k_p * state[2]                                # produce n
        propensities[5] = p.eta * (p.KT - state[4])                       # produce K
        propensities[6] = p.delta * state[1] * state[4]                   # degrade K
    elif model == 'dimeric':
        # state = [R1free, R2free, B1, B2, T, n]
        propensities[0] = p.k_a1 * p.c * state[0]                         # binding R1
        propensities[1] = p.k_d1 * state[2]                               # unbinding B1
        propensities[2] = p.k_a2 * p.c * state[1]                         # binding R2
        propensities[3] = p.k_d2 * state[3]                               # unbinding B2
        propensities[4] = p.k_a3 * state[2] * state[1]                    # binding R2 to B1
        propensities[5] = p.k_d3 * state[4]                               # unbinding T to B1
        propensities[6] = p.k_a4 * state[3] * state[0]                    # binding R1 to B2
        propensities[7] = p.k_d4 * state[4]                               # unbinding T to B2
        propensities[8] = p.k_p * state[4]                                # produce n

    return propensities


# reaction event update dictionary for each model
UPDATE_DICTS = {
    'mode_1': {0: np.array([1.0, 0.0]),  # binding
               1: np.array([-1.0, 0.0]),  # unbinding
               2: np.array([0.0, 1.0])},  # production
    'kpr': {0: np.array([1.0, 0.0, 0.0]),  # binding
            1: np.array([-1.0, 0.0, 0.0]),  # unbinding
            2: np.array([-1.0, 1.0, 0.0]),  # kpr forward step
            3: np.array([0.0, -1.0, 0.0]),  # fall off
            4: np.array([0.0, 0.0, 1.0])},  # produce n
    'adaptive_sorting':
        #			     [0,    1,   2,   n,   K]
            {0: np.array([-1.0, 1.0, 0.0, 0.0, 0.0]),  # ligand binding
             1: np.array([1.0, -1.0, 0.0, 0.0, 0.0]),  # unbinding of ligand from state 1
             2: np.array([0.0, -1.0, 1.0, 0.0, 0.0]),  # kpr forward step
             3: np.array([1.0, 0.0, -1.0, 0.0, 0.0]),  # unbinding of ligand from state 2
             4: np.array([0.0, 0.0, 0.0, 1.0, 0.0]),   # produce n
             5: np.array([0.0, 0.0, 0.0, 0.0, 1.0]),   # produce K
             6: np.array([0.0, 0.0, 0.0, 0.0, -1.0])}, # degrade K
    'dimeric':
        #			 [R1free, R2free, B1, B2,   T,   n]
            {0: np.array([-1.0, 0.0, 1.0, 0.0, 0.0, 0.0]),   # binding R1
             1: np.array([1.0, 0.0, -1.0, 0.0, 0.0, 0.0]),   # unbinding B1
             2: np.array([0.0, -1.0, 0.0, 1.0, 0.0, 0.0]),   # binding R2
             3: np.array([0.0, 1.0, 0.0, -1.0, 0.0, 0.0]),   # unbinding R2
             4: np.array([0.0, -1.0, -1.0, 0.0, 1.0, 0.0]),  # binding R2 to B1
             5: np.array([0.0, 1.0, 1.0, 0.0, -1.0, 0.0]),   # unbinding T to B1
             6: np.array([-1.0, 0.0, 0.0, -1.0, 1.0, 0.0]),  # binding R1 to B2
             7: np.array([1.0, 0.0, 0.0, 1.0, -1.0, 0.0]),   # unbinding T to B2
             8: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])},  # produce n
}


def simulate_traj(num_steps=NUM_STEPS, init_cond=None, model=DEFAULT_MODEL, params=DEFAULT_PARAMS):
    """
    Simulate single trajectory for num_steps
    - model: model string
    - params: params object storing k_on, k_off, c etc
    """
    # prep trajectories
    times = np.zeros(num_steps)
    traj = np.zeros((num_steps, STATE_SIZE[model]))
    # setup init conds
    if init_cond is not None:
        try:
            assert len(init_cond) == STATE_SIZE[model]
        except AssertionError:
            print("len(init_cond) = " + str(len(init_cond)))
            print("STATE_SIZE[model] = " + str(STATE_SIZE[model]))
            assert len(init_cond) == STATE_SIZE[model]
    else:
        init_cond = INIT_CONDS[model]
    traj[0, :] = init_cond
    times[0] = 0.0
    for step in range(num_steps-1):
        # generate two U[0,1] random variables
        r1, r2 = np.random.random(2)
        # generate propensities and their partitions
        alpha = propensities(traj[step], model, params=params)
        alpha_partitions = np.zeros(len(alpha)+1)
        alpha_sum = 0.0
        for i in range(len(alpha)):
            alpha_sum += alpha[i]
            alpha_partitions[i + 1] = alpha_sum
        # find time to first reaction
        tau = np.log(1 / r1) / alpha_sum
        # pick a reaction
        r2_scaled = alpha_sum * r2
        for rxn_idx in range(len(alpha)):
            if alpha_partitions[rxn_idx] <= r2_scaled < alpha_partitions[rxn_idx + 1]:  # i.e. rxn_idx has occurred
                break
        # update state
        traj = update_state(traj, rxn_idx, step, model)
        times[step+1] = times[step] + tau
    return traj, times


def multitraj(num_traj, bound_probabilities, num_steps=NUM_STEPS,
              model=DEFAULT_MODEL, params=DEFAULT_PARAMS,
              init_cond_input=[]):
    """
    Return:
    - traj_array: num_steps x STATE_SIZE[model] x num_traj
    - times_array: num_steps x num_traj
    """
    traj_array = np.zeros((num_steps, STATE_SIZE[model], num_traj), dtype=int)
    times_array = np.zeros((num_steps, num_traj))
    # prep init cond of varying bound states (such that average I.C. matches bound_probabilities)
    if init_cond_input == []:
        init_cond_base = INIT_CONDS[model]
    else:
        init_cond_base = init_cond_input
    draws = np.random.choice(np.arange(0, RECEPTOR_STATE_SIZE[model]), size=num_traj, p=bound_probabilities)

    # simulate k trajectories
    for k in range(num_traj):
        init_vector = np.zeros(RECEPTOR_STATE_SIZE[model])
        init_vector[draws[k]] = 1
        if model in ['adaptive_sorting', 'dimeric']:
            pass  # use the initial conditions given
        else:
            # init_cond_base does not contain unbound state
            init_cond_base[:RECEPTOR_STATE_SIZE[model]-1] = init_vector[1:]
        traj, times = simulate_traj(num_steps=num_steps, init_cond=init_cond_base, model=model, params=params)
        traj_array[:, :, k] = traj
        times_array[:, k] = times

    return traj_array, times_array


if __name__ == '__main__':
    # settings
    model = 'adaptive_sorting'
    num_traj = 50
    num_steps = int(1E4)
    init_bound = np.eye(RECEPTOR_STATE_SIZE[model])[0]

    if model == 'dimeric':
        CUSTOM_PARAMS = DEFAULT_PARAMS
        CUSTOM_PARAMS.k_p = 1E-4
        CUSTOM_PARAMS.c = 6.022E9
    else:
        CUSTOM_PARAMS = DEFAULT_PARAMS
    # compute
    traj_array, times_array = multitraj(num_traj,
                                        bound_probabilities=init_bound,
                                        num_steps=num_steps,
                                        model=model,
                                        params=CUSTOM_PARAMS)

    # plot trajectories
    for k in range(num_traj):
        times_k = times_array[:, k]
        traj_k = traj_array[:, -1, k]
        plt.plot(times_k, traj_k, '--', lw=0.5, alpha=0.5)
    # decorate
    plt.title('Model: %s - %d trajectories' % (model, num_traj))
    plt.xlabel('time')
    plt.ylabel(r'$\langle n_1 \rangle$')
    plt.legend()
    plt.show()
