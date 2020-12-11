from numpy import sqrt


with open("dmudkoffAS.txt", 'r') as file:
    dmudkoffAS_expression = file.readlines()[0]
with open("dNdKOFFAS.py", "w") as f:
    f.write("from numpy import sqrt\n\ndef adaptive_sorting_dNdKOFF_theory(params, times):\n\tc, k_on, k_off, k_p, k_f, alpha, eta, delta, KT, R0 = params.unpack()\n\ttau = 1 / k_off\n\tkappa = k_on\n\treturn [" + dmudkoffAS_expression + " for t in times]\n")
from dNdKOFFAS import adaptive_sorting_dNdKOFF_theory


def adaptive_sorting_meanN_theory(params, times):
    """ These expressions are only true at equilibrium """
    c, k_on, k_off, k_p, k_f, alpha, eta, delta, KT, R0 = params.unpack()
    cstar = eta / delta
    tau = 1 / k_off
    kappa = k_on
    sqrt_term = sqrt(4 * KT * alpha * eta**2 * tau + (eta - KT * alpha * eta * tau + c * R0 * delta * kappa * tau)**2)
    #C0 = (c * R0 * delta * kappa * tau - eta * (1 + KT * alpha * tau) + sqrt_term) / (2 * delta)
    #occ = C0 / (C0 + cstar)
    #zeta = alpha * KT * cstar / k_off
    num = eta + KT * alpha * eta * tau + c * R0 * delta * kappa * tau - sqrt_term
    denom = 2 * delta
    return [k_p * t * num / denom for t in times]


def mode1_meanN_theory(params, times):
    x = params.c * params.k_on / params.k_off
    return [params.k_p * t * x / (1 + x) for t in times]


def mode1_varN_theory(params, times):
    x = params.c * params.k_on / params.k_off
    occ = x / (1 + x)
    term1 = params.k_p * occ
    term2 = 2 * params.k_p**2 * x / (params.k_off * (1 + x)**3)
    return [(term1 + term2) * t for t in times]


def mode1_relKOFF_theory(params, times):
    x = params.c * params.k_on / params.k_off
    const = (1 + x) / x
    const = const * ((1 + x)**2 + 2 * params.k_p / params.k_off)
    return [1 / (params.k_p * t) * const for t in times]


def mode1_dNdKOFF_theory(params, times):
    x = params.c * params.k_on / params.k_off
    const = - (params.k_p / params.k_off) * x / (1 + x)**2
    return [t * const for t in times]


def kpr_meanN_theory(params, times):
    x = params.c * params.k_on / params.k_off
    g = params.k_off / params.k_f
    occ = x / (1 + x)
    return [(1 / (1 + g)) * params.k_p * t * occ for t in times]


def kpr_varN_theory(params, times):
    x = params.c * params.k_on / params.k_off
    g = params.k_off / params.k_f
    occ = x / (1 + x)
    term1 = occ / (1 + g)
    term2 = 2 * params.k_p * x * (1 + g * (2 + x + g * (1 + x)**2)) / (params.k_off * (1 + x)**3 * (1 + g)**3)
    return [(term1 + term2) * params.k_p * t for t in times]


def kpr_relKOFF_theory(params, times):
    x = params.c * params.k_on / params.k_off
    g = params.k_off / params.k_f
    term1 = (1 + g)**2 * params.k_off * (1 + x)**2
    term2 = 2 * params.k_p * (1 + g * (2 + x + g * (1 + x)*2))
    num = (1 + x) * (1 + g) * (term1 + term2)
    denom = params.k_off * params.k_p * x * (1 + g * (2 + x))**2
    return [num / (denom * t) for t in times]


def kpr_dNdKOFF_theory(params, times):
    x = params.c * params.k_on / params.k_off
    g = params.k_off / params.k_f
    num = - params.k_p * x * (1 + g * (2 + x))
    denom = (1 + g)**2 * params.k_off * (1 + x)**2
    return [num * t / denom for t in times]


def dimeric_meanN_theory(params, times):
    K1 = params.k_d1 / params.k_a1
    K2 = params.k_d2 / params.k_a2
    K4 = params.k_d4 / params.k_a4
    IFN = params.c
    Delta = params.R1 - params.R2
    Rt = params.R1 + params.R2
    B = 1 + K4 * (IFN + K2) * (IFN + K1) / (Rt * IFN * K1)
    Teq = (Rt / 2) * (B - sqrt(B**2 - 1 + (Delta / Rt)**2))
    return [Teq * params.k_p * t for t in times]


def dimeric_dNdKOFF_theory(params, times):
    K1 = params.k_d1 / params.k_a1
    K2 = params.k_d2 / params.k_a2
    K4 = params.k_d4 / params.k_a4
    IFN = params.c
    Delta = params.R1 - params.R2
    Rt = params.R1 + params.R2
    ka1 = params.k_a1
    B = 1 + K4 * (IFN + K2) * (IFN + K1) / (Rt * IFN * K1)
    sqrt_term = sqrt(-1 + B**2 + Delta**2 / Rt**2)
    num = (IFN + K2) * (IFN**2 * K4 + IFN * K1 * K4 + IFN * K2 * K4 + K1 * K2 * K4 + IFN * K1 * Rt - IFN * K1 * Rt * sqrt_term)
    denom = 2 * IFN * K1**2 * ka1 * Rt * sqrt_term
    const = num / denom
    return [const * params.k_p * t for t in times]
