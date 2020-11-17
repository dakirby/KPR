def mode1_meanN_theory(params, times):
    return [params.k_p * t * params.x / (1 + params.x) for t in times]


def kpr_meanN_theory(params, times):
    g = params.k_off / params.k_f
    occ = params.x / (1 + params.x)
    return [(1 / (1 + g)) * params.k_p * t * occ for t in times]


def adaptive_sorting_meanN_theory(params, times):
    cstar = params.eta / params.delta
    zeta = params.alpha * params.KT * cstar / params.k_off
    occ = params.c / (params.c + cstar)
    return [zeta * params.k_p * t * occ for t in times]
