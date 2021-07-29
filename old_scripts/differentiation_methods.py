from scipy import interpolate
import numpy as np


def secant_method(record):
    """ The secant method is far too sensitive to stochasticity in data to work
    well for this problem.
    """
    dmu_dkoff = []
    delta_koff_sq = []
    for idx in range(len(record)):
        # can't use secant method on first or last point
        if idx == 0 or idx == len(record) - 1:
            pass
        else:
            dKOFF = record[idx + 1][0] - record[idx][0]
            mu_1 = record[idx + 1][1]
            mu_2 = record[idx - 1][1]
            grad = (mu_1 - mu_2) / (2 * dKOFF)
            dmu_dkoff.append([record[idx][0], grad])

            delta_sq = record[idx][2] / (grad**2)
            delta_koff_sq.append([record[idx][0], delta_sq])
    return dmu_dkoff, delta_koff_sq, None


def spline_method(record, knot_threshold=4):
    """ Splines are supposed to be a more stable way of getting numerical
    derivatives from data. Cubic splines are the typical choice, but 4-th order
    splines seem to produce a more stable estimate of the first derivative in
    my case.
    """
    x = [el[0] for el in record]
    y = [el[1] for el in record]
    var_n = np.array([el[2] for el in record])

    smoothing = np.log10(0.5)
    spline = interpolate.UnivariateSpline(x, y, k=4, s=10**smoothing)
    num_knots = len(spline.get_knots())
    # log schedule for increasing smoothing factor
    while num_knots > knot_threshold:
        if int(smoothing) < int(smoothing + 0.25):
            print("Increasing smoothing to {}".format(10**(smoothing + 0.25)))
        smoothing += 0.25
        spline.set_smoothing_factor(10**smoothing)
        num_knots = len(spline.get_knots())
    grad_func = spline.derivative()
    grad = grad_func(x)
    dmu_dkoff = list(zip(x, grad))
    delta = var_n / np.square(grad)
    delta_koff_sq = list(zip(x, delta))
    return dmu_dkoff, delta_koff_sq, spline


def polynomial_method(record):
    """ A quadratic polynomial should fit the data reasonably well, but the
    derivative approximation becomes significantly worse near the ends of the
    test range.
    """
    x = [el[0] for el in record]
    y = [el[1] for el in record]
    coeffs = np.polyfit(x, y, 2)
    print("Polynomial coefficients: {}".format(coeffs))
    fit = np.poly1d(coeffs)
    grad_func = np.polyder(fit, m=1)
    grad = [grad_func(koff) for koff in x]
    dmu_dkoff = list(zip(x, grad))
    delta = [record[i][2] / (dmu_dkoff[i][1]**2) for i in range(len(dmu_dkoff))]
    delta_koff_sq = list(zip(x, delta))
    return dmu_dkoff, delta_koff_sq, fit


def inverse_polynomial_method(record):
    """ By inspection, the curve may appear like f[x] = 1/x. If this is the case
    then it may be useful to fit a polynomial to the inverse data and then work
    out the derivative of our uninverted function analytically:

    If p[k] is the polynomial fit to the inverse data 1/n[k],
    then -p'[k] * n[k]**2 is the functional form for n'[k]
    """
    x = [el[0] for el in record]
    y = [el[1] for el in record]
    y_inv = [1 / el[1] for el in record]
    coeffs = np.polyfit(x, y_inv, 1)
    print("Polynomial coefficients: {}".format(coeffs))
    fit = np.poly1d(coeffs)
    grad_func = np.polyder(fit, m=1)
    inverse_grad = [grad_func(koff) for koff in x]  # p'[k] from documentation
    dmu_dkoff = [-inverse_grad[i] * y[i]**2 for i in range(len(y))]  # analytic form for the gradient we want
    dmu_dkoff = list(zip(x, dmu_dkoff))
    delta = [record[i][2] / (dmu_dkoff[i][1]**2) for i in range(len(dmu_dkoff))]
    delta_koff_sq = list(zip(x, delta))

    def fit_mean(x):
        """ The function returned should reproduce the mean response data """
        return 1 / fit(x)

    return dmu_dkoff, delta_koff_sq, fit_mean
