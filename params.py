from settings import GLOB_C, GLOB_K_ON, GLOB_K_OFF, GLOB_K_P, GLOB_K_F, GLOB_ALPHA, GLOB_ETA, GLOB_DELTA, GLOB_K0, GLOB_R0
from settings import GLOB_KA1, GLOB_KD1, GLOB_KA2, GLOB_KD2, GLOB_KA3, GLOB_KD3, GLOB_KA4, GLOB_KD4, GLOB_RDIFF

class Params:
    def __init__(self, c=GLOB_C, k_on=GLOB_K_ON, k_off=GLOB_K_OFF,
                 k_p=GLOB_K_P, k_f=GLOB_K_F, alpha=GLOB_ALPHA, eta=GLOB_ETA,
                 delta=GLOB_DELTA, KT=GLOB_K0, R0=GLOB_R0,
                 KA1=GLOB_KA1, KD1=GLOB_KD1, KA2=GLOB_KA2, KD2=GLOB_KD2,
                 KA3=GLOB_KA3, KD3=GLOB_KD3, KA4=GLOB_KA4, KD4=GLOB_KD4,
                 RDIFF=GLOB_RDIFF):
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
        # for dimeric receptor only:
        self.k_a1 = self.k_off
        self.k_d1 = KD1
        self.k_a2 = KA2
        self.k_d2 = KD2
        self.k_a3 = KA3
        self.k_d3 = KD3
        self.k_a4 = KA4
        self.k_d4 = KD4
        self.R1 = (R0 + RDIFF) / 2
        self.R2 = (R0 - RDIFF) / 2

    def unpack(self):
        return self.c, self.k_on, self.k_off, self.k_p, self.k_f, self.alpha, self.eta, self.delta, self.KT, self.R0


DEFAULT_PARAMS = Params()
