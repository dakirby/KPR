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
    return dmu_dkoff, delta_koff_sq


def spline_method(record):
    """ Splines are supposed to be a more stable way of getting numerical
    derivatives from data. However, I find that the cubic spline is overfitting
    my data, which I happen to know is monotonic in koff.
    """
    x = [el[0] for el in record]
    y = [el[1] for el in record]
    spline = interpolate.splrep(x, y)
    grad = interpolate.splev(x, spline, der=1)
    dmu_dkoff = list(zip(x, grad))
    delta = [record[i][2] / (dmu_dkoff[i][1]**2) for i in range(len(dmu_dkoff))]
    delta_koff_sq = list(zip(x, delta))
    return dmu_dkoff, delta_koff_sq


def polynomial_method(record):
    """ A quadratic polynomial should fit the data reasonably well, but the
    derivative approximation becomes significantly worse near the ends of the
    test range.
    """
    x = [el[0] for el in record]
    y = [el[1] for el in record]
    coeffs = np.polyfit(x, y, 3)
    fit = np.poly1d(coeffs)
    grad_func = np.polyder(fit, m=1)
    grad = [grad_func(koff) for koff in x]
    dmu_dkoff = list(zip(x, grad))
    delta = [record[i][2] / (dmu_dkoff[i][1]**2) for i in range(len(dmu_dkoff))]
    delta_koff_sq = list(zip(x, delta))
    return dmu_dkoff, delta_koff_sq
