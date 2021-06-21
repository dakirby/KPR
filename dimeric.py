"""An dimeric receptor is a model of a receptor which comprises two subunits,
implemented here in PySB for stochastic simulation.
Implemented by: Duncan Kirby
"""
from __future__ import print_function
from pysb import *


Model()

Monomer('L', ['r1', 'r2'])
Monomer('R1', ['re', 'ri'])
Monomer('R2', ['re', 'ri'])
Monomer('N', ['s'], {'s': ['u', 'p']})

# Parameter values taken from working IFN model
Parameter('ka1', 3.321155762205247e-14)  # =(2E5 M^-1 s^-1)/(NA*(1E-5 Litres)))
Parameter('kd1', 1)
Parameter('ka2', 4.98173364330787e-13)
Parameter('kd2', 0.015)
Parameter('ka3', 3.623188E-4)
Parameter('kd3', 3E-4)
Parameter('ka4', 3.623188E-4)
Parameter('kd4', 0.3)

Parameter('kp', 1E-6)
Parameter('ku', 1E-3)


Parameter('L_0', 6.022E9)
Parameter('R1_0', 2000)
Parameter('R2_0', 2000)

# Initialization
Initial(L(r1=None,r2=None), L_0)
Initial(R1(re=None, ri=None), R1_0)
Initial(R2(re=None, ri=None), R2_0)

# Observables
Observable('Nobs', N(s='p'))
Observable('Cn', R1(re=1,ri=None)%L(r1=1,r2=2)%R2(re=2,ri=None))
Observable('B1', R1(re=1,ri=None)%L(r1=1,r2=None))
Observable('B2', R2(re=1,ri=None)%L(r1=None,r2=1))

# Rules
Rule('L_bind_R1', R1(re=None,ri=None) + L(r1=None,r2=None) | R1(re=1,ri=None)%L(r1=1,r2=None), ka1, kd1)
Rule('L_bind_R2', R2(re=None,ri=None) + L(r1=None,r2=None) | R2(re=1,ri=None)%L(r1=None,r2=1), ka2, kd2)
Rule('B1_bind_R2', R1(re=1,ri=None)%L(r1=1,r2=None) + R2(re=None,ri=None) | R1(re=1,ri=None)%L(r1=1,r2=2)%R2(re=2,ri=None), ka3, kd3)
Rule('B2_bind_R1', R2(re=2,ri=None)%L(r1=None,r2=2) + R1(re=None,ri=None) | R1(re=1,ri=None)%L(r1=1,r2=2)%R2(re=2,ri=None), ka4, kd4)

Rule('Signal', R1(re=1,ri=None)%L(r1=1,r2=2)%R2(re=2,ri=None) >> R1(re=1,ri=None)%L(r1=1,r2=2)%R2(re=2,ri=None) + N(s='p'), kp)

if __name__ == '__main__':
    print(__doc__, "\n", model)
    print("""
NOTE: This model code is designed to be imported and programatically
manipulated, not executed directly. The above output is merely a
diagnostic aid.""")
