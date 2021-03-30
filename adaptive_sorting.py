"""Adaptive sorting is a model developed by Paul Francois and Gregoire
Altan-Bonnet, implemented here in PySB for stochastic simulation.
Implemented by: Duncan Kirby
"""
from __future__ import print_function
from pysb import *


def intialize_model(init={'R_0': 1E4, 'K_0': 1E3, 'L_0': 1E2}):
    temp = []
    for key, val in init.items():
        temp.append(Parameter(key, val))
    return temp


Model()

Monomer('R', ['s', 'e', 'i'], {'s': ['u', 'p']})    # Receptor
Monomer('K', ['s'], {'s': ['u', 'p']})              # Phosphatase
Monomer('L', ['e'])                                 # ligand
Monomer('N', ['s'], {'s': ['u', 'p']})              # response molecule

# parameter values taken from Lalanne & Francois et al 2013 Phy. Rev. Lett.
# R0 = 1E4, K0 = 1E3, L varies but 1E2 is a typical value
Parameter('kappa', 1e-4)    # association rate
Parameter('koff', 0.1)      # dissociation rate
Parameter('alpha', 3E-4)    # receptor phosphorylation rate
Parameter('b', 0.)          # receptor dephos. rate
Parameter('delta', 1.)      # K deactivation rate const.
Parameter('eps', 1.)        # K activation rate const.
Parameter('kp', 0.)         # N phosphorylation rate
Parameter('ku', 0.)         # N dephos. rate

intialize_model()

Initial(R(e=None, i=None, s='u'), R_0)
Initial(K(s='u'), K_0)
Initial(L(e=None), L_0)

Observable('Nobs', N(s='p'))
Observable('Cn', R(s='p'))

Rule('L_bind_R', L(e=None) + R(e=None, s='u') | L(e=1)%R(e=1, s='u'), kappa, koff)
Rule('Proofread', L(e=1)%R(e=1, s='u') + K(s='u') >> L(e=1)%R(e=1, s='p') + K(s='u'), alpha)
Rule('Reject', L(e=1)%R(e=1, s='p') >> L(e=1)%R(e=1, s='u'), b)
Rule('Fall_off', L(e=1)%R(e=1, s='p') >> L(e=None) + R(e=None, s='u'), koff)
Rule('Activate_K', K(s='p') >> K(s='u'), eps)
Rule('Deactivate_K', K(s='u') + L(e=1)%R(e=1, s='u') >> K(s='p') + L(e=1)%R(e=1, s='u'), delta)
Rule('Signal', N(s='u') + L(e=1)%R(e=1, s='p') >> L(e=1)%R(e=1, s='p') + N(s='p'), kp)
Rule('N_dephos', N(s='p') >> N(s='u'), ku)


if __name__ == '__main__':
    print(__doc__, "\n", model)
    print("""
NOTE: This model code is designed to be imported and programatically
manipulated, not executed directly. The above output is merely a
diagnostic aid.""")
