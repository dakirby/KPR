"""An dimeric receptor is a model of a receptor which comprises two subunits,
implemented here in PySB for stochastic simulation.
Implemented by: Duncan Kirby
"""
from __future__ import print_function
from pysb import *


Model()

Monomer('L', ['r1', 'r2'])
Monomer('R', ['rb'])
Monomer('N', ['s'], {'s': ['u', 'p']})

# Parameter values taken from working IFN model
Parameter('kb', 3.321155762205247e-14)  # =(2E5 M^-1 s^-1)/(NA*(1E-5 Litres)))
Parameter('k_b', 1)
Parameter('kt', 3.623188E-4)
Parameter('k_t', 3E-4)

Parameter('kp', 1E-6)
Parameter('ku', 1E-3)


Parameter('L_0', 6.022E9)
Parameter('R_0', 4000)

# Initialization
Initial(L(r1=None,r2=None), L_0)
Initial(R(rb=None), R_0)

# Observables
Observable('Nobs', N(s='p'))
Observable('Cn', R(rb=1)%L(r1=1,r2=2)%R(rb=2))
Observable('B', R(rb=1)%L(r1=1,r2=None))
Observable('Rf', R(rb=None))

# Rules
Rule('L_bind_R', R(rb=None) + L(r1=None,r2=None) | R(rb=1)%L(r1=1,r2=None), kb, k_b)
Rule('B_bind_R', R(rb=1)%L(r1=1,r2=None) + R(rb=None) | R(rb=1)%L(r1=1,r2=2)%R(rb=2), kt, k_t)

Rule('Signal', R(rb=1)%L(r1=1,r2=2)%R(rb=2) >> R(rb=1)%L(r1=1,r2=2)%R(rb=2) + N(s='p'), kp)

if __name__ == '__main__':
    print(__doc__, "\n", model)
    print("""
NOTE: This model code is designed to be imported and programatically
manipulated, not executed directly. The above output is merely a
diagnostic aid.""")
