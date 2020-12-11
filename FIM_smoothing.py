import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from scipy import interpolate


if __name__ == '__main__':
    smoothing_flag = False
    # Read in data
    output_dir = os.path.join(os.getcwd(), 'output', 'server_output', 'full')
    fname = 'adaptive_sorting_delta_koff_sq.csv'
    path = os.path.join(output_dir, fname)
    data = pd.read_csv(path, index_col=0)
    koff_range = data['koff']
    rel_err = data['delta_koff_sq'] / np.square(koff_range)

    # Fit smoothing function
    knot_threshold = 4
    smoothing = np.log10(0.5)
    spline = interpolate.UnivariateSpline(koff_range, rel_err, k=4, s=10**smoothing)
    num_knots = len(spline.get_knots())
    # log schedule for increasing smoothing factor
    if smoothing_flag:
        while num_knots > knot_threshold:
            if int(smoothing) < int(smoothing + 0.25):
                print("Increasing smoothing to {}".format(10**(smoothing + 0.25)))
            smoothing += 0.25
            spline.set_smoothing_factor(10**smoothing)
            num_knots = len(spline.get_knots())

    # Yield smooth, regularly spaced values for rel_err
    koff_fine = np.logspace(np.log10(min(koff_range)), np.log10(max(koff_range)))
    smoothed_rel_err = spline(koff_fine)
    plt.plot(koff_fine, smoothed_rel_err, 'k--')
    plt.scatter(koff_range, rel_err)
    plt.ylim(0,1)
    plt.xscale('log')
    plt.show()

    df = pd.DataFrame.from_dict({'koff': koff_fine, 'rel_err': smoothed_rel_err})
    df.to_csv(os.path.join(output_dir, fname[:-4] + '_smoothed.csv'))
