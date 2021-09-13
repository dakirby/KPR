"""A KPR receptor is one with additional bound states which do not signal,
implemented here in PySB for stochastic simulation.
Implemented by: Duncan Kirby
"""
from __future__ import print_function
from pysb import *


Model()

Monomer('R', ['e', 'i', 's'], {'s': ['u', 'p1']})   # Receptor
Monomer('L', ['e'])                                 # ligand
Monomer('L_self', ['e'])                                 # self peptide
Monomer('N', ['s'], {'s': ['u', 'p']})              # response molecule

# parameter values taken from Lalanne & Francois et al 2013 Phy. Rev. Lett.
# R0 = 1E4, K0 = 1E3, L varies but 1E2 is a typical value
Parameter('kappa', 1e-4)    # association rate
Parameter('koff', 0.1)      # dissociation rate
Parameter('koff_self', 1.)  # self peptide dissociation rate
Parameter('kf', 0.001)      # proofreading rate
Parameter('kp', 1E-6)         # N phosphorylation rate
Parameter('ku', 1E-3)         # N dephos. rate

Parameter('R_0', 1E4)
Parameter('L_0', 1E2)
Parameter('L_self_0', 0)

Initial(R(e=None, i=None, s='u'), R_0)
Initial(L(e=None), L_0)
Initial(L_self(e=None), L_self_0)

Observable('Nobs', N(s='p'))
Observable('Cn', L(e=1)%R(e=1, s='p1'))
Observable('Cw', L_self(e=1)%R(e=1, s='p1'))

Rule('L_bind_R', L(e=None) + R(e=None, s='u') >> L(e=1)%R(e=1, s='u'), kappa)
Rule('Lself_bind_R', L_self(e=None) + R(e=None, s='u') >> L_self(e=1)%R(e=1, s='u'), kappa)
Rule('Proofreading', L(e=1)%R(e=1, s='u') >> L(e=1)%R(e=1, s='p1'), kf)
Rule('Self_proofreading', L_self(e=1)%R(e=1, s='u') >> L_self(e=1)%R(e=1, s='p1'), kf)
Rule('Fall_off', L(e=1)%R(e=1) >> L(e=None) + R(e=None), koff)
Rule('Self_Fall_off', L_self(e=1)%R(e=1) >> L_self(e=None) + R(e=None), koff_self)

Rule('Signal', L(e=1)%R(e=1, s='p1') >> L(e=1)%R(e=1, s='p1') + N(s='p'), kp)
Rule('Self_Signal', L_self(e=1)%R(e=1, s='p1') >> L_self(e=1)%R(e=1, s='p1') + N(s='p'), kp)


if __name__ == '__main__':
    print(__doc__, "\n", model)
    print("""
NOTE: This model code is designed to be imported and programatically
manipulated, not executed directly. The above output is merely a
diagnostic aid.""")
