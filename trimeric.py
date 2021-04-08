"""An trimeric receptor is a model of a receptor which comprises three subunits,
implemented here in PySB for stochastic simulation.
Implemented by: Duncan Kirby
"""
from __future__ import print_function
from pysb import *


Model()

Monomer('L', ['r1', 'r2', 'r3'])
Monomer('R1', ['re', 'ri'])
Monomer('R2', ['re', 'ri'])
Monomer('R3', ['re', 'ri'])
Monomer('N', ['s'], {'s': ['u', 'p']})

# Parameter values based on IFNAR receptor (for now)
Parameter('ka1', 3.321155762205247e-14)
Parameter('kd1', 1)
Parameter('ka2', 4.98173364330787e-13)
Parameter('kd2', 0.015)
Parameter('ka3', 5.321155762205247e-14)
Parameter('kd3', 0.1)

Parameter('ga12', 3.623188E-4)
Parameter('gd12', 3E-4)
Parameter('ga21', 3.623188E-4)
Parameter('gd21', 3E-4)
Parameter('ga13', 3.623188E-4)
Parameter('gd13', 3E-4)
Parameter('ga31', 3.623188E-4)
Parameter('gd31', 3E-4)
Parameter('ga23', 3.623188E-4)
Parameter('gd23', 3E-4)
Parameter('ga32', 3.623188E-4)
Parameter('gd32', 3E-4)

Parameter('deltaa1', 3.623188E-4)
Parameter('deltad1', 0.006)
Parameter('deltaa2', 3.623188E-4)
Parameter('deltad2', 0.006)
Parameter('deltaa3', 3.623188E-4)
Parameter('deltad3', 0.006)


Parameter('kpa', 1E-3)
Parameter('kpu', 1)


Parameter('L_0', 6.022E9)
Parameter('R1_0', 2000)
Parameter('R2_0', 2000)
Parameter('R3_0', 2000)
Parameter('N_0', 1E4)

# Initialization
Initial(L(r1=None, r2=None, r3=None), L_0)
Initial(R1(re=None, ri=None), R1_0)
Initial(R2(re=None, ri=None), R2_0)
Initial(R3(re=None, ri=None), R3_0)
Initial(N(s='u'), N_0)

# Observables
Observable('Nobs', N(s='p'))
Observable('Cn', R1(re=1,ri=None)%L(r1=1,r2=2,r3=3)%R2(re=2,ri=None)%R3(re=3,ri=None))

# Rules
Rule('L_bind_R1', R1(re=None,ri=None) + L(r1=None,r2=None,r3=None) | R1(re=1,ri=None)%L(r1=1,r2=None,r3=None), ka1, kd1)
Rule('L_bind_R2', R2(re=None,ri=None) + L(r1=None,r2=None,r3=None) | R2(re=1,ri=None)%L(r1=None,r2=1,r3=None), ka2, kd2)
Rule('L_bind_R3', R3(re=None,ri=None) + L(r1=None,r2=None,r3=None) | R3(re=1,ri=None)%L(r1=None,r2=None,r3=1), ka3, kd3)


Rule('B1_bind_R2', R1(re=1,ri=None)%L(r1=1,r2=None,r3=None) + R2(re=None,ri=None) | R1(re=1,ri=None)%L(r1=1,r2=2,r3=None)%R2(re=2,ri=None), ga12, gd12)
Rule('B1_bind_R3', R1(re=1,ri=None)%L(r1=1,r2=None,r3=None) + R3(re=None,ri=None) | R1(re=1,ri=None)%L(r1=1,r2=None,r3=2)%R3(re=2,ri=None), ga13, gd13)

Rule('B2_bind_R1', R2(re=1,ri=None)%L(r1=None,r2=1,r3=None) + R1(re=None,ri=None) | R2(re=1,ri=None)%L(r1=1,r2=2,r3=None)%R2(re=2,ri=None), ga21, gd21)
Rule('B2_bind_R3', R2(re=1,ri=None)%L(r1=None,r2=1,r3=None) + R3(re=None,ri=None) | R2(re=1,ri=None)%L(r1=1,r2=None,r3=2)%R3(re=2,ri=None), ga23, gd23)

Rule('B3_bind_R1', R3(re=1,ri=None)%L(r1=None,r2=None,r3=1) + R1(re=None,ri=None) | R3(re=1,ri=None)%L(r1=2,r2=None,r3=1)%R1(re=2,ri=None), ga31, gd31)
Rule('B3_bind_R2', R3(re=1,ri=None)%L(r1=None,r2=None,r3=1) + R2(re=None,ri=None) | R3(re=1,ri=None)%L(r1=None,r2=2,r3=1)%R2(re=2,ri=None), ga32, gd32)


Rule('T12_bind_R3', R1(re=1,ri=None)%L(r1=1,r2=2,r3=None)%R2(re=2,ri=None) + R3(re=None,ri=None) | R1(re=1,ri=None)%L(r1=1,r2=2,r3=3)%R2(re=2,ri=None)%R3(re=3,ri=None), deltaa3, deltad3)
Rule('T13_bind_R2', R1(re=1,ri=None)%L(r1=1,r2=None,r3=3)%R3(re=3,ri=None) + R2(re=None,ri=None) | R1(re=1,ri=None)%L(r1=1,r2=2,r3=3)%R2(re=2,ri=None)%R3(re=3,ri=None), deltaa2, deltad2)
Rule('T23_bind_R1', R3(re=1,ri=None)%L(r1=None,r2=2,r3=1)%R2(re=2,ri=None) + R1(re=None,ri=None) | R1(re=1,ri=None)%L(r1=1,r2=2,r3=3)%R2(re=2,ri=None)%R3(re=3,ri=None), deltaa1, deltad1)


Rule('Signal', R1(re=1,ri=None)%L(r1=1,r2=2,r3=3)%R2(re=2,ri=None)%R3(re=3,ri=None) + N(s='u') >> R1(re=1,ri=None)%L(r1=1,r2=2,r3=3)%R2(re=2,ri=None)%R3(re=3,ri=None) + N(s='p'), kpa)
Rule('N_dephos', N(s='p') >> N(s='u'), kpu)


if __name__ == '__main__':
    print(__doc__, "\n", model)
    print("""
NOTE: This model code is designed to be imported and programatically
manipulated, not executed directly. The above output is merely a
diagnostic aid.""")
