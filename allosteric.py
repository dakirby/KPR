"""An allosteric receptor is a model of the simplest possible receptor,
implemented here in PySB for stochastic simulation.
Implemented by: Duncan Kirby
"""
from __future__ import print_function
from pysb import *


Model()

Monomer('R', ['e', 'i'])                            # Receptor
Monomer('L', ['e'])                                 # ligand
Monomer('N', ['s'], {'s': ['u', 'p']})              # response molecule

# parameter values taken from Lalanne & Francois et al 2013 Phy. Rev. Lett.
# R0 = 1E4, K0 = 1E3, L varies but 1E2 is a typical value
Parameter('kappa', 1e-4)    # association rate
Parameter('koff', 0.1)      # dissociation rate
Parameter('kp', 1E-6)         # N phosphorylation rate
Parameter('ku', 1E-3)         # N dephos. rate

Parameter('R_0', 1E4)
Parameter('L_0', 1E2)

Initial(R(e=None, i=None), R_0)
Initial(L(e=None), L_0)

Observable('Nobs', N(s='p'))
Observable('Cn', L(e=1)%R(e=1))

Rule('L_bind_R', L(e=None) + R(e=None) | L(e=1)%R(e=1), kappa, koff)
Rule('Signal', L(e=1)%R(e=1) >> L(e=1)%R(e=1) + N(s='p'), kp)


if __name__ == '__main__':
    print(__doc__, "\n", model)
    print("""
NOTE: This model code is designed to be imported and programatically
manipulated, not executed directly. The above output is merely a
diagnostic aid.""")
