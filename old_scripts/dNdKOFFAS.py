from numpy import sqrt

def adaptive_sorting_dNdKOFF_theory(params, times):
	c, k_on, k_off, k_p, k_f, alpha, eta, delta, KT, R0 = params.unpack()
	tau = 1 / k_off
	kappa = k_on
	return [(k_p*t*((KT*alpha*eta - c*R0*delta*kappa)**2 - k_off*(KT*alpha*eta + c*R0*delta*kappa)*(-eta + sqrt((k_off**2*eta**2 + (KT*alpha*eta - c*R0*delta*kappa)**2 + 2*k_off*eta*(KT*alpha*eta + c*R0*delta*kappa))/k_off**2))))/(2.*k_off**3*delta*sqrt((4*k_off*KT*alpha*eta**2 + (k_off*eta - KT*alpha*eta + c*R0*delta*kappa)**2)/k_off**2)) for t in times]
