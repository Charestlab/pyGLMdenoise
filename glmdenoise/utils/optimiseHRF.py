import numpy as np
from glmdenoise.utils.stimMat import constructStimulusMatrices
from glmdenoise.utils import R2_nom_denom
from glmdenoise.utils.get_poly_matrix import get_poly_matrix


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
        thiscol = cnt + list(range(nvols))
        f = f + m1[:, thiscol] * m2[q]
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

    bad = np.all(X == 0, 1)
    good = np.invert(bad)

    # report warning
    if any(bad):
        print(
            "One or more regressors are all zeros; we will estimate a 0 weight for those regressors."
        )

    # do it
    if any(bad):
        f = np.zeros((X.shape[1], X.shape[0]))

        f[good, :] = np.linalg.inv(X[:, good].T * X[:, good]) * X[:, good].T

    else:

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
        design ([type]): [description]
        data ([type]): [description]
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
    numcond, ntime = design[0].shape
    postnumlag = numinhrf - 1
    # precompute for speed
    convdesignpre = []
    for p in range(numruns):
        # expand design matrix using delta functions
        ntime = design[p].shape[0]
        # time x L*conditions
        stimmat = constructStimulusMatrices(
            design[p].T, prenumlag=0, postnumlag=postnumlag
        )

        # time*L x conditions
        convdesignpre.append(stimmat.reshape(numinhrf * ntime, numcond))

    # loop until convergence
    currenthrf = hrfknobs  # initialize
    cnt = 1
    while True:

        # fix the HRF, estimate the amplitudes
        if cnt % 2 == 1:

            # prepare design matrix
            convdesign = []
            for p in range(numruns):

                # convolve original design matrix with HRF
                # number of time points
                convdes = convolveDesign(design[p], currenthrf)

                # project the polynomials out
                convdes = (
                    get_poly_matrix(convdes.shape[0], [0, 1, 2, 3]) * convdes
                )
                # time x conditions

                convdesign.append(convdes)

            # stack design across runs
            stackdesign = np.vstack(convdesign)

            # estimate the amplitudes (output: conditions x voxels)
            currentbeta = mtimesStack(olsmatrix(stackdesign), data)

            # calculate R^2
            modelfit = [convdesign[x] * currentbeta for x in range(numruns)]

            nom_denom = R2_nom_denom(modelfit, data)

            with np.errstate(divide="ignore", invalid="ignore"):
                nom = np.array(nom_denom).sum(0)[0, :]
                denom = np.array(nom_denom).sum(0)[1, :]
                R2 = np.nan_to_num(1 - nom / denom)

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
            iichosen = ii[np.max(1, nii - numforhrf + 1):nii]
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
                ntime = design[p].shape[0]

                # weight and sum based on the current amplitude estimates.
                # only include the good voxels.
                # return shape time*L x voxels
                convdes = convdesignpre[p] * currentbeta[:, hrffitvoxels]

                # remove polynomials and extra regressors
                # time x L*voxels
                convdes = convdes.reshape((ntime, numcond * nhrfvox))
                # time x L*voxels
                convdes = combinedmatrix[p] * convdes
                # time x voxels x L
                convdes = convdes.reshape(ntime, numinhrf, nhrfvox)
                convdesign.append(
                    np.transpose(convdesign[p][:, :, None], (0, 2, 1))
                )

            # estimate the HRF
            previoushrf = currenthrf
            datasubset = np.vstack(
                [data[x][:, hrffitvoxels] for x in range(numruns)]
            )

            stackdesign = np.vstack(convdesign)
            ntime = stackdesign.shape[0]

            stackdesign = stackdesign.reshape((ntime * nhrfvox, numinhrf))

            currenthrf = olsmatrix(stackdesign) * datasubset.Ravel()
            # currenthrf = OLS(stackdesign, datasubset.Ravel())

            # if HRF is all zeros (this can happen when the data are all zeros)
            # get out prematurely
            if np.all(currenthrf == 0):
                break

            # check for convergence
            # how much variance of the previous estimate does
            # the current one explain?
            nom_denom = R2_nom_denom(previoushrf, currenthrf)
            with np.errstate(divide="ignore", invalid="ignore"):
                nom = np.array(nom_denom).sum(0)[0, :]
                denom = np.array(nom_denom).sum(0)[1, :]
                hrfR2 = np.nan_to_num(1 - nom / denom)

            if hrfR2 >= minR2 and cnt > 2:
                break

        cnt = +1

    # sanity check
    # we want to see that we're not converging in a weird place
    # so we compute the coefficient of determination between the
    # current estimate and the seed hrf
    nom_denom = R2_nom_denom(hrfknobs, previoushrf)
    with np.errstate(divide="ignore", invalid="ignore"):
        nom = np.array(nom_denom).sum(0)[0, :]
        denom = np.array(nom_denom).sum(0)[1, :]
        hrfR2 = np.nan_to_num(1 - nom / denom)

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
