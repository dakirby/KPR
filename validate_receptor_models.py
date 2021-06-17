import matplotlib.pyplot as plt
import numpy as np

from adaptive_sorting import model as as_model
from allosteric import model as allo_model
from dimeric import model as dimeric_model
from trimeric import model as trimeric_model
from KPR1 import model as KPR1_model

from pysb_methods import dose_response, response_curve


def T(params, doses):
    K1 = params.kd1.value / params.ka1.value
    K2 = params.kd2.value / params.ka2.value
    K4 = params.kd4.value / params.ka4.value
    Delta = params.R1_0.value - params.R2_0.value
    Rt = params.R1_0.value + params.R2_0.value
    B = 1 + K4 * (doses + K2) * (doses + K1) / (Rt * doses * K1)
    Teq = (Rt / 2) * (B - np.sqrt(np.square(B) - 1 + (Delta / Rt)**2))
    return Teq


def B1(params, doses):
    K1 = params.kd1.value / params.ka1.value
    K2 = params.kd2.value / params.ka2.value
    K4 = params.kd4.value / params.ka4.value
    Delta = params.R1_0.value - params.R2_0.value
    Rt = params.R1_0.value + params.R2_0.value
    IFN = doses
    t1 = -(IFN + K1) * (IFN + K2) * K4 + IFN * K1 * Delta
    t2 = (IFN + K1) * (IFN + K2) * K4 * ((IFN + K1) * (IFN + K2) * K4 + 2 * IFN * K1 * Rt) + np.square(IFN) * K1**2 * Delta**2
    numerator = t1 + np.sqrt(t2)
    denominator = 2 * K1 * (IFN + K1)
    return numerator / denominator


def B2(params, doses):
    K1 = params.kd1.value / params.ka1.value
    K2 = params.kd2.value / params.ka2.value
    K4 = params.kd4.value / params.ka4.value
    Delta = params.R1_0.value - params.R2_0.value
    Rt = params.R1_0.value + params.R2_0.value
    IFN = doses
    t1 = -(IFN + K1) * (IFN + K2) * K4 - IFN * K1 * Delta
    t2 = (IFN + K1) * (IFN + K2) * K4 * ((IFN + K1) * (IFN + K2) * K4 + 2 * IFN * K1 * Rt) + np.square(IFN) * K1**2 * Delta**2
    numerator = t1 + np.sqrt(t2)
    denominator = 2 * K1 * (IFN + K2)
    return numerator / denominator


def dimeric_var_theory(params, doses):
    Rt = params.R1_0.value + params.R2_0.value
    t = T(params, doses)
    b1 = B1(params, doses)
    b2 = B2(params, doses)
    term1 = Rt - 2. * t - b1 - b2
    return 1. / (4. / (term1) + 1./t)


if __name__ == '__main__':
    model_type = 'dimeric'
    plot_dr = False
    plot_rc = False
    plot_sigma = True
    crange = np.logspace(-2, 4, 15) * 1E-12*1E-5*6.022E23
    koffrange = 1 / np.arange(3, 20, 2)
    t_end = 100
    num_traj = 500

    # --------------------------------------------------------------------------
    if model_type == 'allosteric':
        model = allo_model
        n_threshold = 2E2
        koff_name = 'koff'
    elif model_type == 'KPR1':
        model = KPR1_model
        n_threshold = 100.
        koff_name = 'koff'
        crange = crange * 1E-4
    elif model_type == 'adaptive_sorting':
        model = as_model
        n_threshold = 1.
        koff_name = 'koff'
    elif model_type == 'dimeric':
        model = dimeric_model
        n_threshold = 2E2
        koff_name = 'kd4'  # DOES NOT MAINTAIN DETAILED BALANCE
        crange = 100 * crange
    elif model_type == 'trimeric':
        model = trimeric_model
        n_threshold = 2E2
        koff_name = 'kd4'  # DOES NOT MAINTAIN DETAILED BALANCE
        crange = 100 * crange
    else:
        raise NotImplementedError

    if plot_dr:
        y = dose_response(model, crange, 'L_0', t_end, num_traj)
        mean_traj = np.mean(y['Cn'], axis=0)
        std_traj = np.std(y['Cn'], axis=0)

        fig, ax = plt.subplots()
        plt.plot(crange, mean_traj, 'k--', label='simulation')
        ax.fill_between(crange, mean_traj + std_traj, mean_traj - std_traj, 'k', alpha=0.1)
        plt.plot(crange, T(dimeric_model.parameters, crange), 'r--', label='theory')
        plt.xscale('log')
        plt.xlabel('Ligand #')
        plt.ylabel('T')
        plt.legend()
        plt.show()

    if plot_sigma:
        assert model_type == 'dimeric'
        y = dose_response(model, crange, 'L_0', t_end, num_traj)
        mean_traj = np.mean(y['Cn'], axis=0)
        std_traj = np.std(y['Cn'], axis=0)
        theory_var = dimeric_var_theory(dimeric_model.parameters, crange)

        fig, ax = plt.subplots()
        plt.plot(crange, std_traj, 'k--', label='simulation')
        plt.plot(crange, np.sqrt(mean_traj), 'r--', label='low-concentration theory')
        plt.plot(crange, np.sqrt(theory_var), 'b--', label='corrected theory')
        plt.xscale('log')
        plt.xlabel('Ligand #')
        plt.ylabel('Standard Deviation')
        plt.legend()
        plt.show()

    if plot_rc:
        response = response_curve(model, crange, 'L_0', koffrange, koff_name, n_threshold, 'Cn', t_end, num_traj)
        fig, ax = plt.subplots()
        plt.plot(1 / koffrange, response, 'k--')
        plt.yscale('log')
        plt.xlabel(r'$\tau$ (s)')
        plt.ylabel('[Ligand]')
        plt.show()