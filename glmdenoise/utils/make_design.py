import numpy as np
from scipy.interpolate import pchip
from scipy.interpolate import pchip_interpolate
from glmdenoise.utils.normalisemax import normalisemax


def linspacefixeddiff(x, d, n):
    """
    f = linspacefixeddiff(x, d, n)

    Args:
        x ([int]): < x > is a number
        d ([int]): < d > is difference between successive numbers
        n ([type]): < n > is the number of desired points(positive integer)

    Returns:
        a vector of equally spaced values starting at < x > .

    Example:
        assert(linspacefixeddiff(0, 2, 5)==[0, 2, 4, 6, 8])
    """
    x2 = x+d*(n-1)
    return np.linspace(x, x2, n)


def make_design(events, tr, data, hrf):

    # calc
    ntimes = data.shape[0]
    alltimes = linspacefixeddiff(0, tr, ntimes)
    hrftimes = linspacefixeddiff(0, tr, len(hrf))

    # loop over conditions
    conditions = list(set(events.trial_type))
    nconditions = len(conditions)

    # this will be time x conditions
    temp = np.zeros((ntimes, nconditions))
    for i, q in enumerate(conditions):

        # onset times for qth condition in run p
        otimes = events.loc[events['trial_type'] == q, 'onset'].values

        # intialize
        yvals = np.zeros((ntimes))

        # loop over onset times
        for r in otimes:
            # interpolate to find values at the data sampling time points
            sampler = alltimes
            f = pchip(r + hrftimes, hrf, extrapolate=False)(sampler)
            f[np.isnan(f)] = 0
            yvals = yvals + f

        # record
        temp[:, i] = yvals

    return temp
