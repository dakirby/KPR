import numpy as np
import os

# project level
FOLDER_OUTPUT = "output"
if not os.path.exists(FOLDER_OUTPUT):
    os.makedirs(FOLDER_OUTPUT)

# model parameters
# Lalanne & Francois, PRL (2013) provides the following values for adaptive
# sorting in Figure 2:
# kon = 1E-4; R0=1E4; K0=1.0; alpha = 0.3; delta=1.0; eta=1.0
# For stochastic simulation, measure constants in terms of eta and make K0 > 1
GLOB_C = 1E4
GLOB_K_ON = 1E-4
GLOB_K_OFF = 50.0
GLOB_K_P = 10.0
GLOB_K_F = 0.3
GLOB_ETA = 1E-2
GLOB_ALPHA = 0.3
GLOB_DELTA = 1.0

# defined
GLOB_X = GLOB_C * GLOB_K_ON / GLOB_K_OFF
GLOB_PSS_BOUND = GLOB_X / (1 + GLOB_X)

# initial conditions (for single trajectory)
GLOB_N0 = 0.0
GLOB_M0 = 0.0
GLOB_R0 = 1E4
GLOB_K0 = int(1/GLOB_ETA)

# misc
NUM_STEPS = 100

# models
VALID_MODELS = ['mode_1', 'kpr', 'adaptive_sorting']
DEFAULT_MODEL = 'mode_1'

# model structures
NUM_RXN = {'mode_1': 3,
           'kpr': 5,
           'adaptive_sorting': 7}
STATE_SIZE = {'mode_1': 2,
              'kpr': 3,
              'adaptive_sorting': 5}
RECEPTOR_STATE_SIZE = {'mode_1': 2,
                       'kpr': 3,
                       'adaptive_sorting': 3}

# init cond for each model
INIT_CONDS = {'mode_1': [0, GLOB_N0],
              'kpr': [0, 0, GLOB_N0],  # TODO handle init cond for kpr
              'adaptive_sorting': [GLOB_R0, 0, 0, GLOB_N0, GLOB_K0]}
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
             6: np.array([0.0, 0.0, 0.0, 0.0, -1.0])}  # degrade K
}
