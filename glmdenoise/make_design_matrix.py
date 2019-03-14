import numpy as np
from scipy.interpolate import pchip
import pandas as pd


def make_design(events, tr, n_times, hrf=None):
    """generate either a blip design or one convolved with an hrf

    Args:
        events ([type]): [description]
        tr ([type]): [description]
        n_times ([type]): [description]
        hrf ([type], optional): Defaults to None. [description]

    Returns:
        [type]: [description]
    """

    # loop over conditions
    conditions = np.unique(events.trial_type)
    n_conditions = len(set(events['trial_type'].values))

    dm = np.zeros((n_times, n_conditions))

    if hrf is None:
        for i, q in enumerate(conditions):

            # onset times for qth condition in run p
            otimes = np.array(
                events[events['trial_type'] == q]['onset'].values/tr).astype(int)
            yvals = np.zeros((n_times))
            for r in otimes:
                yvals[r] = 1
            dm[:, i] = yvals

    else:
        # calc
        all_times = np.linspace(0, tr*(n_times-1), n_times)
        hrf_times = np.linspace(0, tr*(len(hrf)-1), len(hrf))

        for i, q in enumerate(conditions):
            # onset times for qth condition in run p
            otimes = events.loc[events['trial_type'] == q, 'onset'].values

            # intialize
            yvals = np.zeros((n_times))

            # loop over onset times
            for r in otimes:
                # interpolate to find values at the data sampling time points
                sampler = all_times
                f = pchip(r + hrf_times, hrf, extrapolate=False)(sampler)
                f[np.isnan(f)] = 0
                yvals = yvals + f

            # record
            dm[:, i] = yvals

    return dm