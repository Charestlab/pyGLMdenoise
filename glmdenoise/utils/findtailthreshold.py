import numpy as np
from glmdenoise.utils.robustrange import robustrange
from glmdenoise.utils.picksubset import picksubset
from sklearn.mixture import GaussianMixture as gmdist
import matplotlib.pyplot as plt


def findtailthreshold(v, figpath=None):
    """
     function [f,mns,sds,gmfit] = findtailthreshold(v,wantfig)

     <v> is a vector of values
     <wantfig> (optional) is whether to plot a diagnostic figure. Default: 1.

     Fit a Gaussian Mixture Model (with n=2)
     to the data and find the point that is greater than
     the median and at which the posterior probability
     is equal (50/50) across the two Gaussians.
     This serves as a nice "tail threshold".

     To save on computational load, we take a random subset of
     size 1000000 if there are more than that number of values.
     We also use some discretization in computing our solution.

     return:
     <f> as the threshold
     <mns> as [A B] with the two means (A < B)
     <sds> as [C D] with the corresponding std devs
     <gmfit> with the gmdist object (the order might not
       be the same as A < B)

     example:
     from numpy.random import randn
     f, mns, sds, gmfit = findtailthreshold(np.r_[randn(1000), 5+3*randn(500)], figpath='test.png')
    """

    # internal constants
    numreps = 3  # number of restarts for the GMM
    maxsz = 1000000  # maximum number of values to consider
    nprecision = 500
    # linearly spaced values between median and upper robust range

    # inputs
    if figpath is None:
        wantfig = 0
    else:
        wantfig = 1

    # quick massaging of input
    v = v[np.isfinite(v)]
    if len(v) > maxsz:
        print('warning: too big, so taking a subset')
        v = picksubset(v, maxsz)

    # fit mixture of two gaussians
    gmfit = gmdist(n_components=2, n_init=numreps).fit(v.reshape(-1, 1))

    # figure out a nice range
    rng = robustrange(v.flatten())[0]

    # evaluate posterior
    allvals = np.linspace(np.median(v), rng[1], num=nprecision)
    checkit = gmfit.predict_proba(allvals.reshape(-1, 1))

    # figure out crossing
    np.testing.assert_equal(
        np.any(checkit[:, 0] > .5) and np.any(checkit[:, 0] < .5),
        True,
        err_msg='no crossing of 0.5 detected')
    ix = np.argmin(np.abs(checkit[:, 0]-.5))

    # return it
    f = allvals[ix]

    # prepare other outputs
    mns = gmfit.means_.flatten()
    sds = np.sqrt(gmfit.covariances_.flatten())
    if mns[1] < mns[0]:
        mns = mns[[1, 0]]
        sds = sds[[1, 0]]

    # start the figure
    if wantfig:
        # make figure
        plt.plot(allvals, checkit)
        plt.plot([allvals[ix], allvals[ix]], plt.ylim(), 'k-', linewidth=2)
        plt.title('Posterior Probabilities')
        plt.savefig(figpath)

    return f, mns, sds, gmfit
