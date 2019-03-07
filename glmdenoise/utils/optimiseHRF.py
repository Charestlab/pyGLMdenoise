import numpy as np
from statsmodels.regression.linear_model import OLS
from scipy.signal import convolve2d as conv2
from glmdenoise.utils.stimMat import constructStimulusMatrices
from glmdenoise.utils import R2_nom_denom


def optimiseHRF(design, data, tr, hrfknobs, combinedmatrix, numforhrf=50, hrfthresh=.5, hrffitmask=1):
    """
    [summary]

    Args:
        design ([type]): [description]
        data ([type]): [description]
        tr ([type]): [description]
        hrfknobs ([type]): [description]
        combinedmatrix ([type]): [description]
        numforhrf (int, optional): Defaults to 50. [description]
        hrfthresh (float, optional): Defaults to .5. [description]

    Returns:
        [type]: [description]
    """

    minR2 = .99

    # calc
    numinhrf = len(hrfknobs)
    numruns, numcond, ntime = design.shape()
    postnumlag = numinhrf-1
    # precompute for speed
    convdesignpre = []
    for p in range(numruns):
        # expand design matrix using delta functions
        ntime = design[p].shape[0]
        convdesignpre[p] = constructStimulusMatrices(
            design[p].T, prenumlag=0, postnumlag=postnumlag)  # time x L*conditions
        # time*L x conditions
        convdesignpre[p] = convdesignpre[p].reshape(numinhrf*ntime, numcond)

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
                ntime = design[p].shape[0]
                convdesign[p] = conv2(design[p], currenthrf)     # convolve
                # extract desired subset
                convdesign[p] = convdesign[p][1:ntime, :]

                # remove polynomials and extra regressors
                convdesign[p] = combinedmatrix[p] * \
                    convdesign[p]  # time x conditions

            # stack design across runs
            stackdesign = np.vstack(convdesign)

            # estimate the amplitudes (output: conditions x voxels)
            currentbeta = 0
            currentbeta = [currentbeta+OLS(convdesign[x], data[x])
                           for x in range(numruns)]

            # calculate R^2
            modelfit = [convdesign[x]*currentbeta for x in range(numruns)]

            nom_denom = R2_nom_denom(modelfit, data)

            with np.errstate(divide='ignore', invalid='ignore'):
                nom = np.array(nom_denom).sum(0)[0, :]
                denom = np.array(nom_denom).sum(0)[1, :]
                R2 = np.nan_to_num(1 - nom / denom)

            # figure out indices of good voxels
            if hrffitmask == 1:
                temp = R2
            else:  # if people provided a mask for hrf fitting
                pass
                # temp = copymatrix(R2,~opt.hrffitmask(:),-Inf);
                # shove -Inf in where invalid

            temp[np.isnan(temp)] = -np.inf
            ii = np.argsort(temp)
            nii = len(ii)
            iichosen = ii[np.max(1, nii-numforhrf+1):]
            iichosen = np.setdiff1d(
                iichosen, iichosen[temp[iichosen] == -np.inf]).tolist()
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
                convdesign[p] = convdesignpre[p] * currentbeta[:, hrffitvoxels]

                # remove polynomials and extra regressors
                # time x L*voxels
                convdesign[p] = convdesign[p].reshape((ntime, numcond*nhrfvox))
                # time x L*voxels
                convdesign[p] = combinedmatrix[p]*convdesign[p]
                # time x voxels x L
                convdesign[p] = convdesign[p].reshape(ntime, numinhrf, nhrfvox)
                convdesign[p] = np.transpose(
                    convdesign[p][:, :, None], (0, 2, 1))

            # estimate the HRF
            previoushrf = currenthrf
            datasubset = np.vstack([data[x][:, hrffitvoxels]
                                    for x in range(numruns)])

            stackdesign = np.vstack(convdesign).reshape(
                (ntime*nhrfvox, numinhrf))
            currenthrf = OLS(stackdesign, datasubset.Ravel())

            # if HRF is all zeros (this can happen when the data are all zeros)
            # get out prematurely
            if np.all(currenthrf == 0):
                break

            # check for convergence
            # how much variance of the previous estimate does
            # the current one explain?
            nom_denom = R2_nom_denom(previoushrf, currenthrf)
            with np.errstate(divide='ignore', invalid='ignore'):
                nom = np.array(nom_denom).sum(0)[0, :]
                denom = np.array(nom_denom).sum(0)[1, :]
                hrfR2 = np.nan_to_num(1 - nom / denom)
            if hrfR2 >= minR2 and cnt > 2:
                break

        cnt = + 1

    # sanity check
    # we want to see that we're not converging in a weird place
    # so we compute the coefficient of determination between the
    # current estimate and the seed hrf
    nom_denom = R2_nom_denom(hrfknobs, previoushrf)
    with np.errstate(divide='ignore', invalid='ignore'):
        nom = np.array(nom_denom).sum(0)[0, :]
        denom = np.array(nom_denom).sum(0)[1, :]
        hrfR2 = np.nan_to_num(1 - nom / denom)

    # sanity check to make sure that we are not doing worse.
    if hrfR2 < hrfthresh:
        print('Global HRF estimate is far from the initial seed,'
              'probably indicating low SNR.  We are just going to'
              'use the initial seed as the HRF estimate.\n')
        f = dict()
        f['hrf'] = hrfknobs
        f['hrffitvoxels'] = None
        return f

    # normalize results
    mx = np.max(previoushrf)
    previoushrf = previoushrf / mx
    currentbeta = currentbeta * mx

    # return
    f = dict()
    f['hrf'] = previoushrf
    f['hrffitvoxels'] = hrffitvoxels
    return f
