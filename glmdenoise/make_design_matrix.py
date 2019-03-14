import numpy as np
from scipy.interpolate import pchip
import pandas as pd

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


def make_design(events, tr, ntimes, hrf=None):
    """generate either a blip design or one convolved with an hrf

    Args:
        events ([type]): [description]
        tr ([type]): [description]
        ntimes ([type]): [description]
        hrf ([type], optional): Defaults to None. [description]

    Returns:
        [type]: [description]
    """

    # loop over conditions
    conditions = list(set(events.trial_type))
    nconditions = len(set(events['trial_type'].values))

    dm = np.zeros((ntimes, nconditions))

    if hrf is None:

        for i, q in enumerate(conditions):

            # onset times for qth condition in run p
            otimes = np.array(
                events.loc[events['trial_type'] == q, 'onset'].values/tr).astype(int)
            yvals = np.zeros((ntimes))
            for r in otimes:
                yvals[r] = 1
            dm[:, i] = yvals

    else:
        # calc
        alltimes = linspacefixeddiff(0, tr, ntimes)
        hrftimes = linspacefixeddiff(0, tr, len(hrf))

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
            dm[:, i] = yvals

    return dm