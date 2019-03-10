import numpy as np
from utils.stimMat import constructStimulusMatrices
from utils.R2_nom_denom import R2_nom_denom
from scipy.interpolate import pchip
from pdb import set_trace


def rsquared(y, yhat):
    y = np.array(y)
    yhat = np.array(yhat)

    with np.errstate(divide="ignore", invalid="ignore"):
        nom = np.sum((y - yhat) ** 2, axis=0)
        # denom = np.sum((y - y.mean(axis=0)) ** 2,
        #               axis=0)  # correct denominator
        denom = np.sum(y ** 2, axis=0)  # Kendricks denominator
        rsq = np.nan_to_num(1 - nom / denom)

    # remove inf-values because we might have voxels outside the brain
    rsq[rsq < -1] = -1
    return rsq


def calccodStack(y, yhat):
    """
    [summary]

    Args:
        x ([type]): [description]
        y ([type]): [description]
    """
    numruns = len(y)
    nom = []
    denom = []
    for run in range(numruns):
        with np.errstate(divide="ignore", invalid="ignore"):
            yrun = np.array(y[run])
            yhatrun = np.array(yhat[run])
            nom.append(np.sum((yrun - yhatrun) ** 2, axis=0))
            denom.append(np.sum(yrun ** 2, axis=0))  # Kendricks denominator

    nom = np.array(nom).sum(0)
    denom = np.array(denom).sum(0)

    with np.errstate(divide='ignore', invalid='ignore'):
        f = np.nan_to_num(1 - nom / denom)

    return f


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


def make_design(events, tr, rundata, hrf=None):

    # calc
    ntimes = rundata.shape[0]
    alltimes = linspacefixeddiff(0, tr, ntimes)

    # loop over conditions
    conditions = list(set(events.trial_type))
    nconditions = len(conditions)

    if hrf is None:
        # we need to make a stick matrix
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
                f = np.where(alltimes == min(
                    alltimes, key=lambda x: abs(x-r)))
                yvals[f] = 1

            # record
            temp[:, i] = yvals
    else:
        hrftimes = linspacefixeddiff(0, tr, len(hrf))
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


def mtimesStack(m1, m2):
    """function f = mtimesStack(m1,m2)

    simply return <m1>*np.vstack(m2) but do so in a way that doesn't cause
    too much memory usage.

    Args:
        m1 ([A x B]): is A x B
        m2 ([B x C]): is a stack of matrices such that np.vstack(m2) is B x C
    """
    nruns = len(m2)
    f = 0
    cnt = 0
    for q in range(nruns):
        nvols = m2[q].shape[0]
        thiscol = cnt + np.asarray(list(range(nvols)))
        colrow = m1[:, thiscol] * m2[q]
        f = f + colrow
        cnt = cnt + m2[q].shape[0]

    return f


def olsmatrix(X):
    """
    what we want to do is to perform OLS regression using <X>
    and obtain the parameter estimates.  this is accomplished
    by inv(X'*X)*X'*y = f*y where y is the data (samples x cases).

    what this function does is to return <f> which has dimensions
    parameters x samples.

    we check for a special case, namely, when one or more regressors
    are all zeros.  if we find that this is the case, we issue a warning
    and simply ignore these regressors when fitting.  thus, the weights
    associated with these regressors will be zeros.

    if any warning messages are produced by the inversion process, then we die.
    this is a conservative strategy that ensures that the regression is
    well-behaved (i.e. has a unique, finite solution).  (note that this does
    not cover the case of zero regressors, which is gracefully handled as
    described above.)

    note that no scale normalization of the regressor columns is performed.

    Args:
        X ([2D]): Samples by parameters

    Returns:
        [f]: 2D: parameters by Samples
    """

    bad = np.all(X == 0, axis=0)
    good = np.invert(bad)

    # report warning
    if not np.any(good) == 0:
        print(
            "regressors are all zeros; we will estimate a 0 weight for those regressors."
        )
        f = np.zeros((X.shape[1], X.shape[0]))
        return f

    # do it
    if np.any(bad):
        print(
            "One or more regressors are all zeros; we will estimate a 0 weight for those regressors."
        )
        f = np.zeros((X.shape[1], X.shape[0]))
        X = np.mat(X)
        f[good, :] = np.linalg.inv(X[:, good].T * X[:, good]) * X[:, good].T

    else:
        X = np.mat(X)
        f = np.linalg.inv(X.T * X) * X.T

    return f


def convolveDesign(X, hrf):
    """convolve each column of a 2d design matrix with hrf

    Args:
        X ([2D design matrix]): time by cond
        hrf ([1D hrf function]): hrf

    Returns:
        [convdes]: 2D: Samples by cond
    """
    ntime, ncond = X.shape
    convdes = np.asarray([np.convolve(X[:, x], hrf) for x in range(ncond)]).T

    return convdes[0:ntime, :]


def optimiseHRF(
    design,
    data,
    tr,
    hrfknobs,
    combinedmatrix,
    numforhrf=50,
    hrfthresh=0.5,
    hrffitmask=1,


):
    """
    Optimise hrf from a selection of voxels.
    This uses an iterative fitting optimisation procedure,
    where we fit for betas and then fit for hrf using a fir
    like approach.

    Args:
        events ([type]): [description]
        data ([type]): note that data already had polynomials out
        tr ([type]): [description]
        hrfknobs ([type]): [description]
        combinedmatrix ([type]): [description]
        numforhrf (int, optional): Defaults to 50. [description]
        hrfthresh (float, optional): Defaults to .5. [description]

    Returns:
        [Dict]: we return the hrf and the indices of the v
                oxels used to fit the hrf
    """

    minR2 = 0.99

    # calc
    numinhrf = len(hrfknobs)

    numruns = len(design)

    postnumlag = numinhrf - 1
    # precompute for speed
    convdesignpre = []
    for p in range(numruns):
        events = design[p]
        # construct design matrix
        X = make_design(events, tr, data[p])

        # expand design matrix using delta functions
        ntime, numcond = X.shape
        # time x L*conditions
        stimmat = constructStimulusMatrices(
            X.T, prenumlag=0, postnumlag=postnumlag
        )

        # time*L x conditions
        convdesignpre.append(stimmat.reshape(numinhrf * ntime, numcond))

    # loop until convergence
    currenthrf = hrfknobs  # initialize
    cnt = 1
    while True:
        print(f'\t optimising hrf :{cnt}\n')

        # fix the HRF, estimate the amplitudes
        if cnt % 2 == 1:

            # prepare design matrix
            convdesign = []
            for p in range(numruns):

                # get design matrix with HRF
                # number of time points

                convdes = make_design(design[p], tr, data[p], currenthrf)

                # project the polynomials out
                convdes = combinedmatrix[p] * convdes
                # time x conditions

                convdesign.append(convdes)

            # stack design across runs
            stackdesign = np.vstack(convdesign)

            # estimate the amplitudes (output: conditions x voxels)
            currentbeta = mtimesStack(olsmatrix(stackdesign), data)

            # calculate R^2
            modelfit = [convdesign[p] * currentbeta for p in range(numruns)]

            R2 = calccodStack(modelfit, data)

            # figure out indices of good voxels
            if hrffitmask == 1:
                temp = R2
            else:  # if people provided a mask for hrf fitting
                temp = np.zeros((R2.shape))
                temp[np.invert(hrffitmask.Ravel())] = -np.inf
                # shove -Inf in where invalid

            temp[np.isnan(temp)] = -np.inf
            ii = np.argsort(temp)
            nii = len(ii)
            iichosen = ii[np.max((1, nii - numforhrf)):nii]
            iichosen = np.setdiff1d(
                iichosen, iichosen[temp[iichosen] == -np.inf]
            ).tolist()
            hrffitvoxels = iichosen

        # fix the amplitudes, estimate the HRF
        else:

            nhrfvox = len(hrffitvoxels)

            # prepare design matrix
            convdesign = []
            for p in range(numruns):

                # calc

                # number of time points
                ntime = data[p].shape[0]

                # weight and sum based on the current amplitude estimates.
                # only include the good voxels.
                # return shape time*L x voxels
                convdes = convdesignpre[p] * currentbeta[:, hrffitvoxels]

                # remove polynomials and extra regressors
                # time x L*voxels
                convdes = convdes.reshape(
                    (ntime, -1))
                # time x L*voxels
                convdes = np.array(combinedmatrix[p] * convdes)
                # time x voxels x L
                convdes = convdes.reshape((ntime, numinhrf, nhrfvox))
                convdesign.append(
                    np.transpose(convdes, (0, 2, 1))
                )

            # estimate the HRF
            previoushrf = currenthrf
            datasubset = np.array(np.vstack(
                [data[x][:, hrffitvoxels] for x in range(numruns)]
            ))

            stackdesign = np.vstack(convdesign)
            ntime = stackdesign.shape[0]

            stackdesign = stackdesign.reshape((ntime * nhrfvox, numinhrf))

            currenthrf = olsmatrix(stackdesign) * datasubset.ravel()
            # currenthrf = OLS(stackdesign, datasubset.Ravel())

            # if HRF is all zeros (this can happen when the data are all zeros)
            # get out prematurely
            if np.all(currenthrf == 0):
                print(f'current hrf went all to 0 after {cnt} attempts\n')
                break

            # check for convergence
            # how much variance of the previous estimate does
            # the current one explain?
            hrfR2 = rsquared(previoushrf, currenthrf)

            if hrfR2 >= minR2 and cnt > 2:
                break

        cnt += 1

    # sanity check
    # we want to see that we're not converging in a weird place
    # so we compute the coefficient of determination between the
    # current estimate and the seed hrf
    hrfR2 = rsquared(hrfknobs, previoushrf)

    # sanity check to make sure that we are not doing worse.
    if hrfR2 < hrfthresh:
        print(
            "Global HRF estimate is far from the initial seed,"
            "probably indicating low SNR.  We are just going to"
            "use the initial seed as the HRF estimate.\n"
        )
        f = dict()
        f["hrf"] = hrfknobs
        f["hrffitvoxels"] = None
        return f

    # normalize results
    mx = np.max(previoushrf)
    previoushrf = previoushrf / mx
    currentbeta = currentbeta * mx

    # return
    f = dict()
    f["hrf"] = previoushrf
    f["hrffitvoxels"] = hrffitvoxels
    return f
