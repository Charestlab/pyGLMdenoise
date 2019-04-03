import numpy as np
from glmdenoise.utils.make_design_matrix import make_design


def constructStimulusMatrices(m, prenumlag=0, postnumlag=0, wantwrap=0):
    """construc stimulus matrices from design matrix m

    Args:

        m ([2d matrix]): is a 2D matrix, each row of which is a stimulus
            sequence (i.e. a vector that is all zeros except for ones
            indicating the onset of a given stimulus (fractional values
            are also okay))

        prenumlag (bool or int, optional): Defaults to False. number of
            stimulus points in the past

        postnumlag (bool or int, optional): Defaults to False. number of
            stimulus points in the future

        wantwrap (bool, optional): Defaults to False. whether to wrap
            around
    Returns:
        [2d matrix]: a stimulus matrix of dimensions
            size(m,2) x ((prenumlag+postnumlag+1)*size(m,1)).
            this is a horizontal concatenation of the stimulus
            matrix for the first stimulus sequence, the stimulus
            matrix for the second stimulus sequence, and so on.
            this function is useful for fitting finite impulse
            response (FIR) models.
    """

    # make sure m is numpy
    m = np.asarray(m)

    # get out early
    if not prenumlag and not postnumlag:
        f = m.T
        return f
    else:
        nconds, nvols = m.shape

        # do it
        num = prenumlag + postnumlag + 1
        f = np.zeros((nvols, num*nconds))
        for p in range(nconds):
            i = p + 1
            thiscol = (i - 1) * num + np.array(range(num))
            f[:, thiscol] = constructStimulusMatrix(
                m[p, :], prenumlag, postnumlag, wantwrap
            )

    return f


def constructStimulusMatrix(v, prenumlag, postnumlag, wantwrap=0):
    """Construct stimulus matrix from design vector

    Args:

        v ([1d vector]): v is the stimulus sequence represented as a vector

        prenumlag ([int]): this is the number of stimulus points in the past

        postnumlag ([int]): this is the number of stimulus points in the future

        wantwrap (int, optional): Defaults to 0. whether to wrap around


    Returns:
        [2d array]: return a stimulus matrix of dimensions
            length(v) x (prenumlag+postnumlag+1)
            where each column represents the stimulus at
            a particular time lag.
    """
    v = np.asarray(v)
    total = prenumlag + postnumlag + 1
    f = np.zeros((len(v), total))
    for p in range(total):
        i = p + 1
        if False:
            pass
            # shift = [0 - prenumlag + (p-1)]
            # f[:, p] = np.roll(v, shift, axis=(0, 1)).T
        else:
            temp = -prenumlag + (i - 1)
            if temp < 0:
                pass
                # vindx = range(len(v), 1 - temp)
                # findx = range(len(v)+temp)
                # f[findx, p] = v[vindx]
            else:
                f[temp:, p] = v[: len(v) - temp]
    return f


def calccod(x, y, wantgain=0, wantmeansub=1):
    """Calculate the coefficient of determination

    Args:
        x ([type]): matrix with the same dimensions as y
        y ([type]): matrix with the same dimensions as x
        dim ([type]): is the dimension of interest
        wantgain (int, optional): Defaults to 0. 0 means normal
            1 means allow a gain to be applied to each case of <x>
            to minimize the squared error with respect to <y>.
            in this case, there cannot be any NaNs in <x> or <y>.
            2 is like 1 except that gains are restricted to be non-negative.
            so, if the gain that minimizes the squared error is negative,
            we simply set the gain to be applied to be 0.
            default: 0.
        wantmeansub (int, optional): Defaults to 1.
            0 means do not subtract any mean.  this makes it such that
            the variance quantification is relative to 0.
            1 means subtract the mean of each case of <y> from both
            <x> and <y> before performing the calculation.  this makes
            it such that the variance quantification
            is relative to the mean of each case of <y>.
            note that <wantgain> occurs before <wantmeansub>.
            default: 1.

    calculate the coefficient of determination (R^2) indicating
    the percent variance in <y> that is explained by <x>.  this is achieved
    by calculating 100*(1 - sum((y-x).^2) / sum(y.^2)).  note that
    by default, we subtract the mean of each case of <y> from both <x>
    and <y> before proceeding with the calculation.

    the quantity is at most 100 but can be 0 or negative (unbounded).
    note that this metric is sensitive to DC and scale and is not symmetric
    (i.e. if you swap <x> and <y>, you may obtain different results).
    it is therefore fundamentally different than Pearson's correlation
    coefficient (see calccorrelation.m).

    NaNs are handled gracefully (a NaN causes that data point to be ignored).

    if there are no valid data points (i.e. all data points are
    ignored because of NaNs), we return NaN for that case.

    note some weird cases:
    calccod([],[]) is []

    history:
    2013/08/18 - fix pernicious case where <x> is all zeros and <wantgain>
    is 1 or 2.
    2010/11/28 - add <wantgain>==2 case
    2010/11/23 - changed the output range to percentages.  thus, the range
    is (-Inf,100]. also, we removed the <wantr> input since
    it was dumb.

    example:
    x = np.random.random(100)
    calccod(x,x+0.1*np.random.random(100))
    """

    # input
    dim = np.argmax(x.shape)

    # handle gain
    if wantgain:
        # to get the residuals, we want to do something like y-x*inv(x'*x)*x'*y
        temp = 1/np.dot(x, x) * np.dot(x, y)
        if wantgain == 2:
            # if the gain was going to be negative, rectify it to 0.
            temp[temp < 0] = 0
        x = x * temp

    # propagate NaNs (i.e. ignore invalid data points)
    x[np.isnan(y)] = np.nan
    y[np.isnan(x)] = np.nan

    # handle mean subtraction
    if wantmeansub:
        mn = np.nanmean(y, axis=dim)
        y = y - mn
        x = x - mn

    # finally, compute it
    with np.errstate(divide='ignore', invalid='ignore'):
        nom = np.nansum((y-x) ** 2, axis=dim)
        denom = np.nansum((y**2), axis=dim)
        f = np.nan_to_num(1 - (nom / denom))
    return f


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
        colrow = m1[:, thiscol] @ m2[q]
        f = f + colrow
        cnt = cnt + m2[q].shape[0]

    return f


def olsmatrix(X):
    """OLS regression

    what we want to do is to perform OLS regression using <X>
    and obtain the parameter estimates.  this is accomplished
    by inv(X'\*X)\*X'\*y = f\*y where y is the data (samples x cases).

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
        X (ndarray): Samples by parameters

    Returns:
        (f): 2D parameters by Samples
    """

    bad = np.all(X == 0, axis=0)
    good = np.invert(bad)

    # report warning
    if not np.any(good) == 1:
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
        f[good, :] = np.linalg.inv(
            X[:, good].T  @ X[:, good]) @ X[:, good].T

    else:
        X = np.mat(X)
        f = np.linalg.inv(X.T @ X) @ X.T

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
    """Optimise hrf from a selection of voxels.

    This uses an iterative fitting optimisation procedure,
    where we fit for betas and then fit for hrf using a fir
    like approach.

    Args:
        design (pandas dataframe): this is a pandas data frame with keys:
            ['trial_type']: stimulus condition index
            ['onset']: onsets in s for each event
            ['duration']: duration for each event
        data (2d array): data (time x vox). this data should already
            have polynomials projected out.
        tr (float): the sampling rate in seconds
        hrfknobs (1d array): should be time x 1 with the initial seed
            for the HRF.  The length of this vector indicates the
            number of time points that we will attempt to estimate
            in the HRF. Note on normalization: after fitting the HRF, we
            normalize the HRF to peak at 1 (and adjust amplitudes
            accordingly).
        combinedmatrix (stack of 2d arrays): projection matrix of the
                polynomials and extra regressors (if passed by user).
                This is used to whiten the design matrix.
        numforhrf (int, optional): Defaults to 50.
                is a positive integer indicating the number of voxels
                (with the best R^2 values) to consider in fitting the
                global HRF.  (If there are fewer than that number of
                voxels available, we just use the voxels that are
                available.)
        hrfthresh (float, optional): Defaults to .5.
                If the R^2 between the estimated HRF and the initial HRF
                is less than <hrfthresh>, we decide to just use the initial
                HRF. Set <hrfthresh> to -Inf if you never want to reject
                the estimated HRF.

    Returns:
        (Dict): we return a dictionary with kers:
            ["hrf"]: the optimised hrf (but see note above on hrfthresh)
            ["hrffitvoxels"]: the indices of the voxels used to fit.
            ["convdesign"]: the design convolved with the optimised hrf
            and polynomials projected out.
            ["seedhrf"]: we return the seed hrf for book keeping.

    """

    minR2 = 0.99

    # calc
    numinhrf = len(hrfknobs)

    numruns = len(design)

    postnumlag = numinhrf - 1

    # collect ntimes per run
    ntimes = []

    for p in range(numruns):
        ntimes.append(data[p].shape[0])

    # loop until convergence
    currenthrf = hrfknobs  # initialize
    cnt = 1
    while True:
        print('\t optimising hrf :{}\n'.format(cnt))

        # fix the HRF, estimate the amplitudes
        if cnt % 2 == 1:

            # prepare design matrix
            convdesign = []
            for p in range(numruns):

                # get design matrix with HRF
                # number of time points

                convdes = make_design(design[p], tr, ntimes[p], currenthrf)

                # project the polynomials out
                convdes = np.dot(combinedmatrix[p], convdes)
                # time x conditions

                convdesign.append(convdes)

            # stack design across runs
            stackdesign = np.vstack(convdesign)

            # estimate the amplitudes (output: conditions x voxels)
            currentbeta = mtimesStack(olsmatrix(stackdesign), data)

            # calculate R^2
            modelfit = [np.dot(convdesign[p], currentbeta).astype(np.float32)
                        for p in range(numruns)]

            R2 = calccodStack(data, modelfit)

            # figure out indices of good voxels
            if hrffitmask == 1:
                temp = R2
            else:  # if people provided a mask for hrf fitting
                temp = np.zeros((R2.shape))
                temp[np.invert(hrffitmask.ravel())] = -np.inf
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

                X = make_design(design[p], tr, ntimes[p])

                # expand design matrix using delta functions
                numcond = X.shape[1]
                # time x L*conditions
                stimmat = constructStimulusMatrices(
                    X.T, prenumlag=0, postnumlag=postnumlag
                ).reshape(-1, numcond, order='F').astype(np.float32)

                # calc
                # weight and sum based on the current amplitude estimates.
                # only include the good voxels.
                # return shape time*L x voxels
                convdes = np.dot(
                    stimmat, currentbeta[:, hrffitvoxels]).astype(np.float32)

                # remove polynomials and extra regressors
                # time x L*voxels
                convdes = convdes.reshape(
                    (ntimes[p], -1), order='F')
                # time x L*voxels
                convdes = np.array(np.dot(combinedmatrix[p], convdes))
                # time x voxels x L
                convdes = convdes.reshape((ntimes[p], numinhrf, -1), order='F')
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

            stackdesign = stackdesign.reshape(
                (ntime * nhrfvox, numinhrf), order='F')
            stackdesign = olsmatrix(stackdesign)
            currenthrf = np.asarray(stackdesign.dot(
                datasubset.reshape((-1), order='F')))[0]

            # if HRF is all zeros (this can happen when the data are all zeros)
            # get out prematurely
            if np.all(currenthrf == 0):
                print('current hrf went all to 0 after {} attempts\n'.format(cnt))
                break

            # check for convergence
            # how much variance of the previous estimate does
            # the current one explain?
            hrfR2 = calccod(previoushrf, currenthrf, wantmeansub=0)

            if (hrfR2 >= minR2 and cnt > 2):
                break

        cnt += 1

    # sanity check
    # we want to see that we're not converging in a weird place
    # so we compute the coefficient of determination between the
    # current estimate and the seed hrf
    hrfR2 = calccod(hrfknobs, previoushrf, wantmeansub=0)

    # sanity check to make sure that we are not doing worse.
    if hrfR2 < hrfthresh:
        print(
            "Global HRF estimate is far from the initial seed,"
            "probably indicating low SNR.  We are just going to"
            "use the initial seed as the HRF estimate.\n"
        )
        # prepare design matrix
        convdesign = []
        whitedesign = []
        for p in range(numruns):
            # get design matrix with HRF
            # number of time points
            convdes = make_design(design[p], tr, ntimes[p], hrfknobs)

            # project the polynomials out
            whitedesign.append(np.dot(combinedmatrix[p], convdes))
            # time x conditions

            convdesign.append(convdes)
        f = dict()
        f["hrf"] = hrfknobs
        f["hrffitvoxels"] = None
        f["convdesign"] = convdesign
        f["whitedesign"] = whitedesign
        f["seedhrf"] = hrfknobs
        return f

    # normalize results
    mx = np.max(previoushrf)
    previoushrf = previoushrf / mx
    currentbeta = currentbeta * mx

    # prepare design matrix
    whitedesign = []
    convdesign = []
    for p in range(numruns):
        # get design matrix with HRF
        # number of time points
        convdes = make_design(design[p], tr, ntimes[p], previoushrf)

        # project the polynomials out
        whitedesign.append(np.dot(combinedmatrix[p], convdes))
        # time x conditions

        convdesign.append(convdes)

    # return
    f = dict()
    f["hrf"] = previoushrf
    f["hrffitvoxels"] = hrffitvoxels
    f["convdesign"] = convdesign
    f["whitedesign"] = whitedesign
    f["seedhrf"] = hrfknobs
    f["hrffitmask"] = hrffitmask
    return f
