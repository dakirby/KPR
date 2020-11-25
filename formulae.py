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


def adaptive_sorting_meanN_theory(params, times):
    cstar = params.eta / params.delta
    zeta = params.alpha * params.KT * cstar / params.k_off
    occ = params.c / (params.c + cstar)
    return [params.R0 * zeta * params.k_p * t * occ for t in times]
