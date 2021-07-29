import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

plotting = True

delta_koff = pd.read_csv('output' + os.sep + 'adaptive_sorting_delta_koff_sq.csv', index_col=0)
delta_koff = delta_koff.values

coeffs = np.polyfit(delta_koff[:, 0], delta_koff[:, 1] / np.square(delta_koff[:, 0]), 3)
print("Polynomial coefficients: {}".format(coeffs))
fit = np.poly1d(coeffs)

fine_koff_axis = np.arange(min(delta_koff[:, 0]), max(delta_koff[:, 0]), 0.0125)

if plotting:
    plt.scatter(delta_koff[:, 0], delta_koff[:, 1] / np.square(delta_koff[:, 0]))
    plt.plot(fine_koff_axis, fit(fine_koff_axis), 'k--')
    plt.show()

record = np.zeros((len(fine_koff_axis), 2))
record[:, 0] = fine_koff_axis
record[:, 1] = fit(fine_koff_axis)
record = pd.DataFrame(record, columns=['koff', 'rel_err_koff'])
print(record)
record.to_csv('output' + os.sep + 'adaptive_sorting_smoothed_delta_koff_sq.csv')
