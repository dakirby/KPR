from settings import GLOB_C, GLOB_K_ON, GLOB_K_OFF, GLOB_K_P, GLOB_K_F, GLOB_ALPHA, GLOB_ETA, GLOB_DELTA, GLOB_K0, GLOB_R0


class Params:
    def __init__(self, c=GLOB_C, k_on=GLOB_K_ON, k_off=GLOB_K_OFF,
                 k_p=GLOB_K_P, k_f=GLOB_K_F, alpha=GLOB_ALPHA, eta=GLOB_ETA,
                 delta=GLOB_DELTA, KT=GLOB_K0, R0=GLOB_R0):
        self.c = c
        self.k_on = k_on
        self.k_off = k_off
        self.k_p = k_p
        self.k_f = k_f
        self.alpha = alpha
        self.eta = eta
        self.delta = delta
        self.KT = KT
        self.R0 = R0

    def unpack(self):
        return self.c, self.k_on, self.k_off, self.k_p, self.k_f, self.alpha, self.eta, self.delta, self.KT, self.R0


DEFAULT_PARAMS = Params()
