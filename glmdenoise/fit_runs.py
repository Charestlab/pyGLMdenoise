import numpy as np
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)


def fit_runs(data, design):
    """Fits a least square of combined runs.

    The matrix addition is equivalent to concatenating the list of data and
    the list of design and fit it all at once. However, this is more memory
    efficient.

    Arguments:
        runs {list} -- List of runs. Each run is an TR x voxel sized array
        DM {list} -- List of design matrices. Each design matrix
                     is an TR x predictor sizec array

    Returns:
        [array] -- betas from fit

    TODO handle zero regressors.
    """

    X = np.vstack(design)
    X = np.linalg.inv(X.T @ X) @ X.T

    betas = 0
    start_col = 0

    for run in range(len(data)):
        n_vols = data[run].shape[0]
        these_cols = np.arange(n_vols) + start_col
        betas += X[:, these_cols] @ data[run]
        start_col += data[run].shape[0]

    return betas
