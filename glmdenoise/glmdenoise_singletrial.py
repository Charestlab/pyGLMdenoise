import os
import numpy as np
from glmdenoise.utils.normalisemax import normalisemax
from glmdenoise.utils.gethrf import getcanonicalhrf, getcanonicalhrflibrary
from glmdenoise.utils.chunking import chunking as ch
from glmdenoise.fit_runs import fit_runs
from glmdenoise.utils.make_design_matrix import make_design
from glmdenoise.cross_validate import cross_validate
import matplotlib.pyplot as plt
from glmdenoise.utils.make_image_stack import make_image_stack
from glmdenoise.utils.optimiseHRF import convolveDesign
from glmdenoise.utils.findtailthreshold import findtailthreshold
from glmdenoise.whiten_data import whiten_data
from glmdenoise.utils.optimiseHRF import (mtimesStack,
                                          olsmatrix,
                                          optimiseHRF,
                                          calccod,
                                          calccodStack)

dir0 = os.path.dirname(os.path.realpath(__file__))


def GLMestimatesingletrial(
        design,
        data,
        stimdur,
        tr,
        outputdir=None,
        opt=None
        ):
    """

       <design> is the experimental design. There are two possible cases:
       1. A where A is a matrix with dimensions time x conditions.
          Each column should be zeros except for ones indicating condition
          onsets.
       2. {A1 A2 A3 ... An} where each of the A's are like the previous case.
          The different A's correspond to different runs, and different runs
          can have different numbers of time points. However, all A's must have
          the same number of conditions.
       Note that we ultimately compute single-trial response estimates (one
       estimate for each condition onset), and these will be provided in
       chronological order. However, by specifying that a given condition
       occurs more than one time over the course of the experiment, this
       information can and will be used for cross-validation purposes.

     <data> is the time-series data with dimensions X x Y x Z x time or a
       cell vector of elements that are each X x Y x Z x time. XYZ can be
       collapsed such that the data are given as a 2D matrix (units x time),
       which is useful for surface-format data.
       The dimensions of <data> should mirror that of <design>. For example,
       <design> and <data> should have the same number of runs, the same number
       of time points, etc.
       <data> should not contain any NaNs. We automatically convert <data> to
       single format if not already in single format.
     <stimdur> is the duration of a trial in seconds. For example, 3.5 means
       that you expect the neural activity from a given trial to last
       for 3.5 s.
     <tr> is the sampling rate in seconds. For example, 1 means that we get a
       new time point every 1 s. Note that <tr> applies to both <design>
       and <data>.
     <outputdir> (optional) is a directory to which files will be written.
       (If the directory does not exist, we create it; if the directory already
       exists, we delete its contents so we can start fresh.) If you set
       <outputdir> to None, we will not create a directory and no files will
       be written.
       Default is 'GLMestimatesingletrialoutputs' (created in the current
       working directory).
     <opt> (optional) is a struct with the following optional fields:

       *** MAJOR, HIGH-LEVEL FLAGS ***

       <wantlibrary> (optional) is
         0 means use an assumed HRF
         1 means determine the best HRF for each voxel using the
           library-of-HRFs approach
         Default: 1.
       <wantglmdenoise> (optional) is
         0 means do not perform GLMdenoise
         1 means perform GLMdenoise
         Default: 1.
       <wantfracridge> (optional) is
         0 means do not perform ridge regression
         1 means perform ridge regression
         Default: 1.
       <chunknum> (optional) is the number of voxels that we will process at
         the same time. This number should be large in order to speed
         computation, but should not be so large that you run out of RAM.
         Default: 50000.
       <xvalscheme> (optional) is a cell vector of vectors of run indices,
         indicating the cross-validation scheme. For example, if we have 8
         runs, we could use [[1, 2], [3, 4], [5, 6], [7, 8]] which indicates
         to do 4 folds of cross-validation, first holding out the 1st and 2nd
         runs, then the 3rd and 4th runs, etc.
         Default: {[1] [2] [3] ... [n]} where n is the number of runs.
       <sessionindicator> (optional) is 1 x n (where n is the number of runs)
        with positive integers indicating the run groupings that are
        interpreted as "sessions". The purpose of this input is to allow for
        session-wise z-scoring of single-trial beta weights for the purposes
        of hyperparameter evaluation. Note that the z-scoring has effect only
        INTERNALLY: it is used merely to calculate the cross-validation
        performance and the associated hyperparameter selection; the outputs
        of this function do not reflect z-scoring, and the user may wish to
        apply z-scoring. Default: 1*ones(1,n) which means to interpret allruns
        as coming from the same session.

       *** I/O FLAGS ***

       <wantfileoutputs> (optional) is a logical vector [A B C D] indicating
         which of the four model types to save to disk (assuming that they
         are computed).
         A = 0/1 for saving the results of the ONOFF model
         B = 0/1 for saving the results of the FITHRF model
         C = 0/1 for saving the results of the FITHRF_GLMDENOISE model
         D = 0/1 for saving the results of the FITHRF_GLMDENOISE_RR model
         Default: [1 1 1 1] which means save all computed results to disk.
       <wantmemoryoutputs> (optional) is a logical vector [A B C D] indicating
         which of the four model types to return in the output <results>. The
         user must be careful with this, as large datasets can require a lot of
         RAM. If you do not request the various model types, they will be
         cleared from memory (but still potentially saved to disk).
         Default: [0 0 0 1] which means return only the final type-D model.

       *** GLM FLAGS ***

       <extraregressors> (optional) is time x regressors or a cell vector
         of elements that are each time x regressors. The dimensions of
         <extraregressors> should mirror that of <design> (i.e. same number of
         runs, same number of time points). The number of extra regressors
         does not have to be the same across runs, and each run can have zero
         or more extra regressors. If [] or not supplied, we do
         not use extra regressors in the model.
       <maxpolydeg> (optional) is a non-negative integer with the maximum
         polynomial degree to use for polynomial nuisance functions, which
         are used to capture low-frequency noise fluctuations in each run.
         Can be a vector with length equal to the number of runs (this
         allows you to specify different degrees for different runs).
         Default is to use round(L/2) for each run where L is the
         duration in minutes of a given run.
       <wantpercentbold> (optional) is whether to convert amplitude estimates
         to percent BOLD change. This is done as the very last step, and is
         accomplished by dividing by the absolute value of 'meanvol' and
         multiplying by 100. (The absolute value prevents negative values in
         'meanvol' from flipping the sign.) Default: 1.

       *** HRF FLAGS ***

       <hrftoassume> (optional) is time x 1 with an assumed HRF that
         characterizes the evoked response to each trial. We automatically
         divide by the maximum value so that the peak is equal to 1. Default
         is to generate a canonical HRF (see getcanonicalhrf.m).
         Note that the HRF supplied in <hrftoassume> is used in only two
         instances:
         (1) it is used for the simple ONOFF type-A model, and (2) if the
             user sets <wantlibrary> to 0, it is also used for the type-B,
             type-C, and type-D models.
       <hrflibrary> (optional) is time x H with H different HRFs to choose
         from for the library-of-HRFs approach. We automatically normalize
         each HRF to peak at 1.
         Default is to generate a library of 20 HRFs (see
         getcanonicalhrflibrary).
         Note that if <wantlibrary> is 0, <hrflibrary> is clobbered with the
         contents of <hrftoassume>, which in effect causes a single assumed
         HRF to be used.

       *** MODEL TYPE A (ONOFF) FLAGS ***

       (none)

       *** MODEL TYPE B (FITHRF) FLAGS ***

       <wantlss> (optional) is 0/1 indicating whether 'least-squares-separate'
         estimates are desired. If 1, then the type-B model will be estimated
         using the least-squares-separate method (as opposed to ordinary
         least squares). Default: 0.

       *** MODEL TYPE C (FITHRF_GLMDENOISE) FLAGS ***

       <numpcstotry> (optional) is a non-negative integer indicating the
         maximum number of PCs to enter into the model. Default: 10.
       <brainthresh> (optional) is [A B] where A is a percentile for voxel
         intensity values and B is a fraction to apply to the percentile. These
         parameters are used in the selection of the noise pool.
         Default: [99 0.1].
       <brainR2> (optional) is an R^2 value (percentage). After fitting the
         type-A model, voxels whose R^2 is below this value are allowed to
         enter the noise pool.
         Default is [] which means to automatically determine a good value.
       <brainexclude> (optional) is X x Y x Z (or XYZ x 1) with 1s indicating
         voxels to specifically exclude when selecting the noise pool. 0 means
         all voxels can be potentially chosen. Default: 0.
       <pcR2cutoff> (optional) is an R^2 value (percentage). To decide the
         number of PCs to include, we examine a subset of the available voxels.
         Specifically, we examine voxels whose type-A model R^2 is above
         <pcR2cutoff>. Default is []
         which means to automatically determine a good value.
       <pcR2cutoffmask> (optional) is X x Y x Z (or XYZ x 1) with 1s
         indicating all possible voxels to consider when selecting the subset
         of voxels. 1 means all voxels can be potentially selected. Default: 1.
       <pcstop> (optional) is
         A: a number greater than or equal to 1 indicating when to stop adding
            PCs into the model. For example, 1.05 means that if the
            cross-validation performance with the current number of PCs is
            within 5 of the maximum observed, then use that number of PCs.
            (Performance is measured relative to the case of 0 PCs.) When
            <pcstop> is 1, the selection strategy reduces to simply choosing
            the PC number that achieves the maximum. The advantage of stopping
            early is to achieve a selection strategy that is robust to noise
            and shallow performance curves and that avoids overfitting.
        -B: where B is the number of PCs to use for the final model. B can be
            any integer between 0 and opt.numpcstotry. Note that if -B case is
            used, cross-validation is NOT performed for the type-C model, and
            instead weblindly use B PCs.
         Default: 1.05.

       *** MODEL TYPE D (FITHRF_GLMDENOISE_RR) FLAGS ***

       <fracs> (optional) is a vector of fractions that are greater than 0
         and less than or equal to 1. We automatically sort in descending
         order and ensure the fractions are unique. These fractions indicate
         the regularization levels to evaluate using fractional ridge
         regression (fracridge) and cross-validation.
         Default: fliplr(.05:.05:1).
         A special case is when <fracs> is specified as a single scalar value.
         In this case, cross-validation is NOT performed for the type-D model,
         and we instead blindly usethe supplied fractional value for the type-D
         model.
       <wantautoscale> (optional) is whether to automatically scale and offset
         the model estimates from the type-D model to best match the
         unregularized estimates. Default: 1.

     This function computes up to four model outputs (called type-A (ONOFF),
     type-B (FITHRF), type-C (FITHRF_GLMDENOISE), and type-D
     (FITHRF_GLMDENOISE_RR)),and either saves the model outputs to disk,
     or returns them in <results>, or both,depending on what the user
     specifies.

     There are a variety of cases that you can achieve. Here are some examples:

     - wantlibrary=1, wantglmdenoise=1, wantfracridge=1 [Default]
         A = simple ONOFF model
         B = single-trial estimates using a tailored HRF for every voxel
         C = like B but with GLMdenoise regressors added into the model
         D = like C but with ridge regression regularization (tailored to each
             voxel)

     - wantlibrary=0
         A fixed assumed HRF is used in all model types.

     - wantglmdenoise=0, wantfracridge=0
         Model types C and D are not computed.

     - wantglmdenoise=0, wantfracridge=1
         Model type C is not computed; model type D is computed using 0
         GLMdenoise regressors.

     - wantglmdenoise=1, wantfracridge=0
         Model type C is computed; model type D is not computed.

     - wantlss=1
         Model type B is computed, but using least-squares-separate instead of
         OLS.Other model types, if computed, use OLS.

     Note that if you set wantglmdenoise=1, you MUST have repeats of
     conditions andan associated cross-validation scheme (<opt.xvalscheme>),
     UNLESS you specify opt.pcstop = -B. In other words, you can perform
     wantglmdenoise without any cross-validation, but you need to provide
     opt.pcstop = -B.

     Note that if you set wantfracridge=1, you MUST have repeats of conditions
     andan associated cross-validation scheme (<opt.xvalscheme>), UNLESS you
     specify a single scalar opt.fracs. In other words, you can perform
     wantfracridge without any cross-validation, but you need to provide
     opt.fracs as a scalar.

     OUTPUTS:

     There are various outputs for each of the four model types:

     <modelmd> is either
       (1) the HRF (time x 1) and ON-OFF beta weights (X x Y x Z)
       (2) the full set of single-trial beta weights (X x Y x Z x TRIALS)
     <R2> is model accuracy expressed in terms of R^2 (percentage).
     <R2run> is R2 separated by run
     <meanvol> is the mean of all volumes
     <FitHRFR2> is the R2 for each of the different HRFs in the library
     <FitHRFR2run> is separated by run
     <HRFindex> is the 1-index of the best HRF
     <HRFindexrun> is HRFindex separated by run
     <noisepool> indicates voxels selected for the noise pool
     <pcregressors> indicates the full set of candidate GLMdenoise regressors
                    that were found
     <glmbadness> is cross-validation results for GLMdenoise
     <pcvoxels> is the set of voxels used to summarize GLMdenoise
                cross-validation results
     <xvaltrend> is the summary GLMdenoise cross-validation result on which
                pcnum selection is done
     <pcnum> is the number of PCs that were selected for the final model
     <FRACvalue> is the fractional regularization level chosen for each voxel
     <scaleoffset> is the scale and offset applied to RR estimates to best
                   match the unregularized result

     History:
     - 2020/08/22 - Implement opt.sessionindicator. Also, now the cross-validation units now reflect
                    the "session-wise z-scoring" hyperparameter selection approach; thus, the cross-
                    validation units have been CHANGED relative to prior analyses!
     - 2020/05/14 - Version 1.0 released!
                    (Tweak some documentation; output more results; fix a small
                    bug (opt.fracs(1)~=1).)
     - 2020/05/12 - Add pcvoxels output.
     - 2020/05/12 - Initial version. Beta version. Use with caution.
    """

    # DEAL WITH INPUTS

    # massage <design> and sanity-check it
    if type(design) is not list:
        design = [design]

    numcond = design[0].shape[1]
    for p in range(len(design)):
        np.testing.assert_array_equal(
            np.unique(design),
            [0, 1],
            err_msg='<design> must consist of 0s and 1s')
        condmsg = \
            'all runs in <design> should have the same number of conditions'
        np.testing.assert_equal(
            design[p].shape[1],
            numcond,
            err_msg=condmsg)
        # if the design happened to be a sparse?
        # design[p] = np.full(design[p])

    # massage <data> and sanity-check it
    if type(data) is not list:
        data = [data]

    # make sure it is single
    for p in range(len(data)):
        data[p] = data[p].astype(np.float32, copy=False)

    np.testing.assert_equal(
        np.all(np.isfinite(data[0].flatten())),
        True,
        err_msg=f'We checked the first run and '
        'found some non-finite values (e.g. NaN, Inf).'
        'Please fix and re-run.')
    np.testing.assert_equal(
        len(design),
        len(data),
        err_msg='<design> and <data> should have '
        'the same number of runs')

    # calc
    numruns = len(design)
    is3d = data[0].ndim > 2  # is this the X Y Z T case?
    if is3d:
        wantfig = 1
        x, y, z, _ = data[0].shape
        for p in range(len(data)):
            data[p] = np.moveaxis(data[p], -1, 0)
            data[p] = data[p].reshape([data[p].shape[0], -1])

    else:
        print('assuming that your data has shape nfeatures x nconditions')

    dimdata = 1  # how many of the first dimensions have data
    dimtime = 0  # the dimension with time points
    numvoxels = data[0].shape[dimdata]
    nx = numvoxels

    # check number of time points and truncate if necessary
    for p in range(len(data)):
        if data[p].shape[dimtime] > design[p].shape[0]:
            print(
                f'WARNING: run {p} has more time points'
                'in <data> than <design>. We are truncating'
                '<data>.\n')
            data[p] = data[p][:, range(design.shape[0])]

        if data[p].shape[dimtime] < design[p].shape[0]:
            print(
                f'WARNING: run {p} has more time points in <design>'
                'than <data>. We are truncating <design>.\n')
            design[p] = design[p][range(data[p].shape[0]), :]

    # inputs
    if opt is None:
        opt = {}

    if 'wantlibrary' not in opt:
        opt['wantlibrary'] = 1

    if 'wantglmdenoise' not in opt:
        opt['wantglmdenoise'] = 1

    if 'wantfracridge' not in opt:
        opt['wantfracridge'] = 1

    if 'chunknum' not in opt:
        opt['chunknum'] = 50000

    if 'xvalscheme' not in opt:
        opt['xvalscheme'] = range(numruns)

    if 'sessionindicator' not in opt:
        opt['sessionindicator'] = np.ones((1, numruns))

    if 'wantfileoutputs' not in opt:
        opt['wantfileoutputs'] = [1, 1, 1, 1]

    if 'wantmemoryoutputs' not in opt:
        opt['wantmemoryoutputs'] = [0, 0, 0, 1]

    if 'extraregressors' not in opt:
        opt['extraregressors'] = []  # IC deal with this later

    if 'maxpolydeg' not in opt:
        opt['maxpolydeg'] = [round(
            ((data[p].shape[dimtime]*tr)/60)/2) for p in range(numruns)]

    if 'wantpercentbold' not in opt:
        opt['wantpercentbold'] = 1

    if 'hrftoassume' not in opt:
        opt['hrftoassume'] = normalisemax(
          getcanonicalhrf(stimdur, tr),
          dim='global')

    if 'hrflibrary' not in opt:
        opt['hrflibrary'] = getcanonicalhrflibrary(stimdur, tr).T

    if 'wantlss' not in opt:
        opt['wantlss'] = 0

    if 'numpcstotry' not in opt:
        opt['numpcstotry'] = 10

    if 'brainthresh' not in opt:
        opt['brainthresh'] = [99, 0.1]

    if 'brainR2' not in opt:
        opt['brainR2'] = []

    if 'brainexclude' not in opt:
        opt['brainexclude'] = 0

    if 'pcR2cutoff' not in opt:
        opt['pcR2cutoff'] = []

    if 'pcR2cutoffmask' not in opt:
        opt['pcR2cutoffmask'] = 1

    if 'pcstop' not in opt:
        opt['pcstop'] = 1.05

    if 'fracs' not in opt:
        opt['fracs'] = np.linspace(1, 0.05, 20)

    if 'wantautoscale' not in opt:
        opt['wantautoscale'] = 1

    # deal with length issues and other miscellaneous things
    if type(opt['maxpolydeg']) is int:
        opt['maxpolydeg'] = np.tile(opt['maxpolydeg'], numruns).tolist()

    opt['hrftoassume'] = normalisemax(opt['hrftoassume'], dim='global')
    opt['hrflibrary'] = normalisemax(opt['hrflibrary'], 0)
    opt['fracs'] = np.unique(opt['fracs'])[::-1]
    np.testing.assert_equal(
        np.all(opt['fracs'] > 0),
        True,
        err_msg='fracs must be greater than 0')

    np.testing.assert_equal(
        np.all(opt['fracs'] <= 1),
        True,
        err_msg='fracs must be less than or equal to 1')

    if outputdir is not None and is3d:
        wantfig = 1  # if outputdir is not None, we want figures

    # deal with output directory
    if outputdir is None:
        cwd = os.getcwd()
        outputdir = os.path.join(cwd, 'GLMestimatesingletrialoutputs')

    if os.path.exists(outputdir):
        os.removedirs(outputdir)
        os.makedirs(outputdir)
    else:
        os.makedirs(outputdir)

    if np.any(opt['wantfileoutputs']):
        errm = 'you must specify an <outputdir> in order to get file outputs'
        np.testing.assert_equal(
            type(outputdir),
            str,
            err_msg=errm)

    # deal with special library stuff
    if opt['wantlibrary'] == 0:
        opt['hrflibrary'] = opt['hrftoassume']

    # calc
    nh = opt['hrflibrary'].shape[1]

    # figure out chunking scheme
    chunks = ch(
        range(nx), int(np.ceil(nx/np.ceil(numvoxels/opt['chunknum']))))

    # deal with special cases
    if opt['wantglmdenoise'] == 1:
        errm = '<wantglmdenoise> is 1, but you didnt request type C nor type D'
        np.testing.assert_equal(
            np.any(
                opt['wantfileoutputs'][-2:]
                ) or np.any(opt['wantmemoryoutputs'][-2:]),
            True,
            err_msg=errm)

    if opt['wantfracridge'] == 1:
        np.testing.assert_equal(
            opt['wantfileoutputs'][3] == 1 or opt['wantmemoryoutputs'][3] == 1,
            True,
            err_msg='<wantfracridge> is 1, but you did not request type D')

    if opt['wantlss'] == 1:
        np.testing.assert_equal(
            opt['wantfileoutputs'][1] == 1 or opt['wantmemoryoutputs'][1] == 1,
            True,
            err_msg='<wantlss> is 1, but you did not request type B')

    # initialize output
    results = []

    # PRE-PROCESSING FOR THE EXPERIMENTAL DESIGN

    # calculate the number of trials
    # number of trials in each run
    numtrialrun = np.asarray([np.sum(x.flatten()) for x in design])
    numtrials = np.sum(numtrialrun).astype(int)  # number of total trials

    # create a single-trial design matrix and calculate a bunch
    # of extra information
    designSINGLE = []
    cnt = 0

    # 1 x numtrials indicating which condition each trial belongs to
    stimorder = []

    # each element is the vector of trial indices associated with the run
    validcolumns = []

    # each element is the vector of actual condition numbers occurring
    # with a given run
    stimix = []

    # loop through runs
    for p in range(len(design)):
        designSINGLE.append(np.zeros((design[p].shape[0], numtrials)))

        run_validcolumns = []
        # loop through the volumes for that run
        for q in range(design[p].shape[0]):
            # if a condition was presented on that volume
            # find which
            temp = np.where(design[p][q, :])[0]

            # if that volume had a condition shown
            if not np.size(temp) == 0:
                # flip it on
                designSINGLE[p][q, cnt] = 1
                # keep track of the order
                stimorder.append(temp[0])
                run_validcolumns.append(cnt)
                cnt += 1
        validcolumns.append(run_validcolumns)

        stimix.append(np.asarray(stimorder)[np.asarray(run_validcolumns)])

    # FIT TYPE-A MODEL [ON-OFF]

    # The approach:
    # (1) Every stimulus is treated as the same.
    # (2) We assume the HRF.

    # define
    whmodel = 1

    # collapse all conditions and fit
    print('*** FITTING TYPE-A MODEL (ONOFF) ***\n')
    design0 = [np.sum(x, axis=1) for x in design]

    # convolve the single design with the hrf
    convdes = []
    for run_i, run in enumerate(design0):
        X = convolveDesign(run, opt['hrftoassume'])
        convdes.append(X[:, np.newaxis])

    results0 = {}
    onoffR2 = cross_validate(
        data,
        convdes,
        opt['extraregressors'],
        poly_degs=range(opt['maxpolydeg'][p]))[0]
    meanvol = np.vstack(data).mean(0)

    # save to disk if desired
    if opt['wantfileoutputs'][whmodel] == 1:
        file0 = os.path.join(outputdir, 'TYPEA_ONOFF.npy')
        print(f'*** Saving results to {file0}. ***\n')
        np.save(file0, onoffR2, meanvol)

    # figures
    if wantfig and is3d:
        """ TODO
          port normalizerange.m and add to makeimstack
        """
        plt.imshow(
            make_image_stack(np.reshape(onoffR2, [x, y, z])), cmap='hot')
        plt.colorbar()
        plt.savefig(os.path.join(outputdir, 'onoffR2.png'))
        plt.close('all')
        plt.imshow(
            make_image_stack(np.reshape(meanvol, [x, y, z])), cmap='gray')
        plt.colorbar()
        plt.savefig(os.path.join(outputdir, 'meanvol.png'))
        plt.close('all')

    # preserve in memory if desired, and then clean up
    if opt['wantmemoryoutputs'][whmodel] == 1:
        results = {}
        results['onoffR2'] = onoffR2
        results['meanvol'] = meanvol

    # DETERMINE THRESHOLDS
    if wantfig:
        thresh = findtailthreshold(
            onoffR2.flatten(),
            os.path.join(outputdir, 'onoffR2hist.png'))[0]
    else:
        thresh = findtailthreshold(onoffR2.flatten())[0]

    if 'brainR2' not in opt:
        opt['brainR2'] = thresh

    if 'pcR2cutoff' not in opt:
        opt['pcR2cutoff'] = thresh

    # FIT TYPE-B MODEL [FITHRF]

    # The approach:
    # (1) Fit single trials.
    # (2) Evaluate the library of HRFs (or the single assumed HRF).
    #     Choose based on R2 for each voxel.

    # if the user does not want file output nor memory output AND
    # if the number of HRFs to choose
    # from is just 1, we can short-circuit this whole endeavor!
    if opt['wantfileoutputs'][1] == 0 and \
            opt['wantmemoryoutputs'][1] == 0 and \
            opt['hrflibrary'].shape[1] == 1:

        # short-circuit all of the work
        HRFindex = np.ones((nx, ny, nz))  # easy peasy

    else:

        # define
        whmodel = 2

        # initialize

        # X x Y x Z x HRFs with R2 values (all runs)
        FitHRFR2 = []
        # X x Y x Z x runs x HRFs with R2 separated by runs
        FitHRFR2run = []
        modelmd = []

        # loop over chunks
        print('*** FITTING TYPE-B MODEL (FITHRF) ***\n')
        for i, chunk in enumerate(chunks):
            print(f'working on chunk {i} of {len(chunks)}.\n')

            data_chunk = [dat[:, chunk] for dat in data]

            # do the fitting and accumulate all the betas
            # someX x Y x Z x trialbetas x HRFs

            FitHRFR2_t = []
            FitHRFR2run_t = []
            betas_t = []
            for p in range(opt['hrflibrary'].shape[1]):

                # convolve single design with current hrf
                current_hrf = opt['hrflibrary'][:, p]
                conv_designSINGLE = [convolveDesign(
                    X, current_hrf) for X in designSINGLE]

                wdata, wdesign = whiten_data(data_chunk, conv_designSINGLE)
                modelfits = [((olsmatrix(x, verbose=False) @ y).T @ x.T).T for x, y in zip(
                    wdesign, data_chunk)]

                FitHRFR2_t.append(calccodStack(data_chunk, modelfits))

                r2run = [calccod(
                    cdata,
                    mfit,
                    0, 0, 0) for cdata, mfit in zip(wdata, modelfits)]

                FitHRFR2run_t.append(np.stack(r2run))

                betas_t.append(fit_runs(wdata, wdesign, verbose=False))

            # reshape fits
            FitHRFR2_t = np.stack(FitHRFR2_t)  # n_hrfs by xchunk by y by z
            FitHRFR2run_t = np.stack(FitHRFR2run_t)
            # n_hrfs by xchunk by y by z by run?
            betas_t = np.stack(betas_t)
            # n_hrfs by xchunk by y by z x trialbetas

            # keep only the betas we want
            ii = np.argmax(FitHRFR2_t, axis=0)
            betas = []
            for jj in range(len(chunk)):
                this_betas = betas_t[:, :, jj]
                betas.append(this_betas[ii[jj], :])

            # append
            FitHRFR2.append(FitHRFR2_t)
            FitHRFR2run.append(FitHRFR2run_t)
            modelmd.append(betas)

        FitHRFR2 = np.hstack(FitHRFR2)
        FitHRFR2run = np.hstack(FitHRFR2run)
        modelmd = np.hstack(modelmd)

        # use R2 to select the best HRF for each voxel
        R2 = np.max(FitHRFR2, axis=0)  # R2 is X x Y x Z
        HRFindex = np.argmax(FitHRFR2, axis=0)  # HRFindex is X x Y x Z
        # also, use R2 from each run to select best HRF
        HRFindexrun = np.argmax(FitHRFR2run, axis=0)

        # using each voxel's best HRF, what are the corresponding R2run values?
        R2run = FitHRFR2run[HRFindexrun, :, :, :, :]  # R2run is X x Y x Z x runs

        # FIT TYPE-B MODEL (LSS) INTERLUDE BEGIN

        # if opt.wantlss, we have to use the determined HRFindex and re-fit the entire dataset
        # using LSS estimation. this will simply replace 'modelmd' with a new version.
        # so that's what we have to do here.

        if opt['wantlss']:

              # initalize
              modelmd = np.zeros(nx*ny*nz,numtrials).astype(np.float32)     # X*Y*Z x trialbetas  [the final beta estimates]

              # loop over chunks
              print('*** FITTING TYPE-B MODEL (FITHRF but with LSS estimation) ***\n')
              for chunk in range(len(chunks)):
                  print(f'working on chunk {z} of {len(chunks)}.\n')
            
                  # loop over possible HRFs
                  for hh in range(opt['hrflibrary'].shape[1]):
              
                      # figure out which voxels to process.
                      # this will be a vector of indices into the small chunk that we are processing.
                      # our goal is to fully process this set of voxels!
                      goodix = np.where(HRFindex[chunks[z],:,:]==hh).flatten()
                  
                      # extract the data we want to process.
                      data0 = [dat[chunks[z][goodix, ]]]
                      data0 = cellfun(@(x) subscript(squish(x(chunks{z},:,:,:),3),{goodix ':'}),data,'UniformOutput',0);
                    
                      % calculate the corresponding indices relative to the full volume
                      temp = zeros(size(HRFindex));
                      temp(chunks{z},:,:) = 1;
                      relix = subscript(find(temp),goodix);
                
                      % define options
                      optA = struct('extraregressors',{opt.extraregressors}, ...
                                    'maxpolydeg',opt.maxpolydeg, ...
                                    'wantpercentbold',opt.wantpercentbold, ...
                                    'suppressoutput',1);

                      % do the GLM
                      cnt = 0;
                      for rrr=1:length(designSINGLE)  % each run
                        for ccc=1:numtrialrun(rrr)    % each trial
                          designtemp = designSINGLE{rrr};
                          designtemp = [designtemp(:,cnt+ccc) sum(designtemp(:,setdiff(1:size(designtemp,2),cnt+ccc)),2)];
                          [results0,cache] = GLMestimatemodel(designtemp,data0{rrr}, ...
                                                      stimdur,tr,'assume',opt.hrflibrary(:,hh),0,optA);
                          modelmd(relix,cnt+ccc) = results0.modelmd{2}(:,1);
                          clear results0;
                        end
                        cnt = cnt + numtrialrun(rrr);
                      end
              
                end
              
              
        
          % deal with dimensions
          modelmd = reshape(modelmd,[nx ny nz numtrials]);

        end

      %% %%%%%%%% FIT TYPE-B MODEL (LSS) INTERLUDE END

      % save to disk if desired
      allvars = {'FitHRFR2','FitHRFR2run','HRFindex','HRFindexrun','R2','R2run','modelmd','meanvol'};
      if opt.wantfileoutputs(whmodel)==1
        file0 = fullfile(outputdir,'TYPEB_FITHRF.mat');
        fprintf('*** Saving results to %s. ***\n',file0);
        save(file0,allvars{:},'-v7.3');
      end

      % figures?
      if wantfig && is3d
        imwrite(uint8(255*makeimagestack(HRFindex,[1 nh])),jet(256),fullfile(outputdir,'HRFindex.png'));
      end

      % preserve in memory if desired, and then clean up
      if opt.wantmemoryoutputs(whmodel)==1
        results{whmodel} = struct;
        for p=1:length(allvars)
          results{whmodel}.(allvars{p}) = eval(allvars{p});
        end
      end
      clear FitHRFR2 FitHRFR2run R2 R2run modelmd;  % Note that we keep HRFindex and HRFindexrun around!!

    end

    %% %%%%%%%%%%%%%%%%%%% COMPUTE GLMDENOISE REGRESSORS

    % if the user does not want to perform GLMdenoise, we can just skip all of this
    if opt.wantglmdenoise==0

      % just create placeholders
      pcregressors = {};
      noisepool = [];

    else

      % figure out the noise pool
      thresh = prctile(meanvol(:),opt.brainthresh(1))*opt.brainthresh(2);    % threshold for non-brain voxels
      bright = meanvol > thresh;                                             % logical indicating voxels that are bright (brain voxels)
      badR2 = onoffR2 < opt.brainR2;                                         % logical indicating voxels with poor R2
      noisepool = bright & badR2 & ~opt.brainexclude;                        % logical indicating voxels that satisfy all criteria

      % determine noise regressors
      pcregressors = {};
      fprintf('*** DETERMINING GLMDENOISE REGRESSORS ***\n');
      for p=1:length(data)

        % extract the time-series data for the noise pool
        temp = subscript(squish(data{p},dimdata),{find(noisepool) ':'})';  % time x voxels

        % project out polynomials from the data
        temp = projectionmatrix(constructpolynomialmatrix(size(temp,1),0:opt.maxpolydeg(p))) * temp;

        % unit-length normalize each time-series (ignoring any time-series that are all 0)
        [temp,len] = unitlengthfast(temp,1);
        temp = temp(:,len~=0);

        % perform SVD and select the top PCs
        [u,s,v] = svds(double(temp*temp'),opt.numpcstotry);
        u = bsxfun(@rdivide,u,std(u,[],1));  % scale so that std is 1
        pcregressors{p} = cast(u,'single');

      end
      clear temp len u s v;

    end

    %% %%%%%%%%%%%%%%%%%%% CROSS-VALIDATE TO FIGURE OUT NUMBER OF GLMDENOISE REGRESSORS

    % if the user does not want GLMdenoise, just set some dummy values
    if opt.wantglmdenoise==0
      pcnum = 0;
      xvaltrend = [];
      glmbadness = [];
      pcvoxels = [];

    % in this case, the user decides (and we can skip the cross-validation)
    elseif opt.pcstop <= 0
      pcnum = -opt.pcstop;
      xvaltrend = [];
      glmbadness = [];
      pcvoxels = [];

    % otherwise, we have to do a lot of work
    else
      
      % initialize
      glmbadness = zeros(nx*ny*nz,1+opt.numpcstotry,'single');    % X * Y * Z x 1+npc  [squared beta error for different numbers of PCs]
      
      % loop over chunks
      fprintf('*** CROSS-VALIDATING DIFFERENT NUMBERS OF REGRESSORS ***\n');
      for z=1:length(chunks)
        fprintf('working on chunk %d of %d.\n',z,length(chunks));

        % loop over possible HRFs
        for hh=1:size(opt.hrflibrary,2)
      
          % figure out which voxels to process.
          % this will be a vector of indices into the small chunk that we are processing.
          % our goal is to fully process this set of voxels!
          goodix = flatten(find(HRFindex(chunks{z},:,:)==hh));
          
          % extract the data we want to process.
          data0 = cellfun(@(x) subscript(squish(x(chunks{z},:,:,:),3),{goodix ':'}),data,'UniformOutput',0);
          
          % calculate the corresponding indices relative to the full volume
          temp = zeros(size(HRFindex));
          temp(chunks{z},:,:) = 1;
          relix = subscript(find(temp),goodix);
      
          % perform GLMdenoise
          clear results0;
          for ll=0:opt.numpcstotry
      
            % define options
            optA = struct('maxpolydeg',opt.maxpolydeg, ...
                          'wantpercentbold',0, ...
                          'suppressoutput',1);
            optA.extraregressors = cell(1,length(data0));
            if ll>0
              for rr=1:length(data0)
                optA.extraregressors{rr} = cat(2,optA.extraregressors{rr},pcregressors{rr}(:,1:ll));
              end
            end
            
            % do the GLM
            [results0(ll+1),cache] = GLMestimatemodel(designSINGLE,data0, ...
                                      stimdur,tr,'assume',opt.hrflibrary(:,hh),0,optA);
      
            % save some memory
            results0(ll+1).models = [];
            results0(ll+1).modelse = [];
          
          end
      
          % compute the cross-validation performance values
          glmbadness(relix,:) = calcbadness(opt.xvalscheme,validcolumns,stimix,results0,opt.sessionindicator);  % voxels x regularization levels
          clear results0;
      
        end
        
      end

      % compute xvaltrend
      ix = find((onoffR2(:) > opt.pcR2cutoff) & (opt.pcR2cutoffmask(:)));  % vector of indices
      if isempty(ix)
        fprintf('Warning: no voxels passed the pcR2cutoff and pcR2cutoffmask criteria. Using the best 100 voxels.\n');
        if isequal(opt.pcR2cutoffmask,1)
          ix2 = find(ones(size(onoffR2)));
        else
          ix2 = find(opt.pcR2cutoffmask==1);
        end
        assert(length(ix2) > 0,'no voxels are in pcR2cutoffmask');
        [~,ix3] = sort(onoffR2(ix2),'descend');
        num = min(100,length(ix2));
        ix = ix2(ix3(1:num));
      end
      xvaltrend = -median(glmbadness(ix,:),1);  % NOTE: sign flip so that high is good
      assert(all(isfinite(xvaltrend)));

      % create for safe-keeping
      pcvoxels = logical(zeros(nx,ny,nz));
      pcvoxels(ix) = 1;
      
      % choose number of PCs
      chosen = 0;  % this is the fall-back
      curve = xvaltrend - xvaltrend(1);  % this is the performance curve that starts at 0 (corresponding to 0 PCs)
      mx = max(curve);                   % store the maximum of the curve
      best = -Inf;                       % initialize (this will hold the best performance observed thus far)
      for p=0:opt.numpcstotry
      
        % if better than best so far
        if curve(1+p) > best
      
          % record this number of PCs as the best
          chosen = p;
          best = curve(1+p);
        
          % if we are within opt.pcstop of the max, then we stop.
          if best*opt.pcstop >= mx
            break;
          end
        
        end
      
      end
      
      % record the number of PCs
      pcnum = chosen;
      
      % deal with dimensions
      glmbadness = reshape(glmbadness,nx,ny,nz,[]);
      
    end

    %% %%%%%%%%%%%%%%%%%%% FIT TYPE-C + TYPE-D MODELS [FITHRF_GLMDENOISE, FITHRF_GLMDENOISE_RR]

    % setup
    todo = [];
    if opt.wantglmdenoise==1 && (opt.wantfileoutputs(3)==1 || opt.wantmemoryoutputs(3)==1)
      todo = [todo 3];  % the user wants the type-C model returned
    end
    if opt.wantfracridge==1 && (opt.wantfileoutputs(4)==1 || opt.wantmemoryoutputs(4)==1)
      todo = [todo 4];  % the user wants the type-D model returned
    end

    % process models
    for ttt=1:length(todo)
      whmodel = todo(ttt);

      %% we need to do some tricky setup
      
      % if this is just a GLMdenoise case, we need to fake it
      if whmodel==3
        fracstouse = [1];    % here, we need to fake this in order to get the outputs
        fractoselectix = 1;
        autoscaletouse = 0;  % not necessary, so turn off
      end
      
      % if this is a fracridge case
      if whmodel==4
      
        % if the user specified only one fraction
        if length(opt.fracs)==1
        
          % if the first one is 1, this is easy
          if opt.fracs(1)==1
            fracstouse = [1];
            fractoselectix = 1;
            autoscaletouse = 0;  % not necessary, so turn off
    
          % if the first one is not 1, we might need 1
          else
            fracstouse = [1 opt.fracs];
            fractoselectix = 2;
            autoscaletouse = opt.wantautoscale;
          end
        
        % otherwise, we have to do costly cross-validation
        else

          % set these
          fractoselectix = NaN;
          autoscaletouse = opt.wantautoscale;
        
          % if the first one is 1, this is easy
          if opt.fracs(1)==1
            fracstouse = opt.fracs;
          
          % if the first one is not 1, we might need 1
          else
            fracstouse = [1 opt.fracs];
          end

        end
        
      end

      %% ok, proceed

      % initialize
      modelmd =     zeros(nx*ny*nz,numtrials,'single');     % X * Y * Z x trialbetas  [the final beta estimates]
      R2 =          zeros(nx,ny,nz,'single');               % X x Y x Z               [the R2 for the specific optimal frac]
      R2run =       zeros(nx*ny*nz,numruns,'single');       % X * Y * Z x runs        [the R2 separated by runs for the optimal frac]
      FRACvalue   = zeros(nx,ny,nz,'single');               % X x Y x Z               [best fraction]
      if isnan(fractoselectix)
        rrbadness   = zeros(nx*ny*nz,length(opt.fracs),'single');   % X x Y x Z       [rr cross-validation performance]
      else
        rrbadness = [];
      end
      scaleoffset = zeros(nx*ny*nz,2,'single');             % X * Y * Z x 2           [scale and offset]

      % loop over chunks
      if whmodel==3
        fprintf('*** FITTING TYPE-C MODEL (GLMDENOISE) ***\n');
      else
        fprintf('*** FITTING TYPE-D MODEL (GLMDENOISE_RR) ***\n');
      end
      for z=1:length(chunks)
        fprintf('working on chunk %d of %d.\n',z,length(chunks));

        % loop over possible HRFs
        for hh=1:size(opt.hrflibrary,2)

          % figure out which voxels to process.
          % this will be a vector of indices into the small chunk that we are processing.
          % our goal is to fully process this set of voxels!
          goodix = flatten(find(HRFindex(chunks{z},:,:)==hh));
        
          % extract the data we want to process.
          data0 = cellfun(@(x) subscript(squish(x(chunks{z},:,:,:),3),{goodix ':'}),data,'UniformOutput',0);
        
          % calculate the corresponding indices relative to the full volume
          temp = zeros(size(HRFindex));
          temp(chunks{z},:,:) = 1;
          relix = subscript(find(temp),goodix);

          % process each frac
          clear results0;
          for ll=1:length(fracstouse)

            % define options
            optA = struct('maxpolydeg',opt.maxpolydeg, ...
                          'wantpercentbold',0, ...
                          'suppressoutput',1, ...
                          'frac',fracstouse(ll));
            optA.extraregressors = cell(1,length(data0));
            if pcnum > 0
              for rr=1:length(data0)
                optA.extraregressors{rr} = cat(2,optA.extraregressors{rr},pcregressors{rr}(:,1:pcnum));
              end
            end

            % fit the entire dataset using the specific frac
            [results0(ll),cache] = GLMestimatemodel(designSINGLE,data0, ...
                                    stimdur,tr,'assume',opt.hrflibrary(:,hh),0,optA);
          
            % save some memory
            results0(ll).models = [];
            results0(ll).modelse = [];
        
          end
          
          % perform cross-validation if necessary
          if isnan(fractoselectix)
            
            % compute the cross-validation performance values
            rrbadness0 = calcbadness(opt.xvalscheme,validcolumns,stimix,results0,opt.sessionindicator);

            % this is the weird special case where we have to ignore the artificially added 1
            if opt.fracs(1) ~= 1
              [~,FRACindex0] = min(rrbadness0(:,2:end),[],2);
              FRACindex0 = FRACindex0 + 1;
              rrbadness(relix,:) = rrbadness0(:,2:end);
            else
              [~,FRACindex0] = min(rrbadness0,[],2);  % pick best frac (FRACindex0 is V x 1 with the index of the best frac)
              rrbadness(relix,:) = rrbadness0;
            end

          % if we already know fractoselectix, skip the cross-validation
          else
            FRACindex0 = fractoselectix*ones(length(relix),1);
          end
        
          % prepare output
          FRACvalue(relix) = fracstouse(FRACindex0);
          for ll=1:length(fracstouse)
            ii = find(FRACindex0==ll);  % indices of voxels that chose the llth frac
          
            % scale and offset to match the unregularized result
            if autoscaletouse
              for vv=1:length(ii)
                X = [results0(ll).modelmd{2}(ii(vv),:); ones(1,numtrials)]';
                h = olsmatrix(X)*results0(1).modelmd{2}(ii(vv),:)';  % Notice the 1
                if h(1) < 0
                  h = [1 0]';
                end
                scaleoffset(relix(ii(vv)),:) = h;
                modelmd(relix(ii(vv)),:) = X*h;
              end
            else
              scaleoffset = [];
              modelmd(relix(ii),:) = results0(ll).modelmd{2}(ii,:);
            end
          
            R2(relix(ii))        = results0(ll).R2(ii);
            R2run(relix(ii),:)   = results0(ll).R2run(ii,:);
          end

        end

      end

      % deal with dimensions
      modelmd = reshape(modelmd,[nx ny nz numtrials]);
      modelmd = bsxfun(@rdivide,modelmd,abs(meanvol)) * 100;  % deal with percent BOLD change
      R2run = reshape(R2run,[nx ny nz numruns]);
      if ~isempty(scaleoffset)
        scaleoffset = reshape(scaleoffset,[nx ny nz 2]);
      end
      if isnan(fractoselectix)
        rrbadness = reshape(rrbadness,nx,ny,nz,[]);
      end
      
      % save to disk if desired
      if whmodel==3
        allvars = {'HRFindex','HRFindexrun','glmbadness','pcvoxels','pcnum','xvaltrend', ...
                  'noisepool','pcregressors','modelmd','R2','R2run','meanvol'};
        file0 = fullfile(outputdir,'TYPEC_FITHRF_GLMDENOISE.mat');
      else
        allvars = {'HRFindex','HRFindexrun','glmbadness','pcvoxels','pcnum','xvaltrend', ...
                  'noisepool','pcregressors','modelmd','R2','R2run','rrbadness','FRACvalue','scaleoffset','meanvol'};
        file0 = fullfile(outputdir,'TYPED_FITHRF_GLMDENOISE_RR.mat');
      end
      if opt.wantfileoutputs(whmodel)==1
        save(file0,allvars{:},'-v7.3');
      end

      % figures?
      if wantfig
        if whmodel==3
          if is3d
            imwrite(uint8(255*makeimagestack(noisepool,[0 1])),gray(256),fullfile(outputdir,'noisepool.png'));
            imwrite(uint8(255*makeimagestack(pcvoxels, [0 1])),gray(256),fullfile(outputdir,'pcvoxels.png'));
          end
          figureprep;
          plot(0:opt.numpcstotry,xvaltrend);
          straightline(pcnum,'v','r-');
          xlabel('Number of GLMdenoise regressors');
          ylabel('Cross-validation performance (higher is better)');
          figurewrite('xvaltrend',[],[],outputdir);
        end
        if whmodel==4 && is3d
          imwrite(uint8(255*makeimagestack(R2,[0 100]).^0.5),hot(256),fullfile(outputdir,'typeD_R2.png'));
          imwrite(uint8(255*makeimagestack(FRACvalue,[0 1])),copper(256),fullfile(outputdir,'FRACvalue.png'));
        end
      end

      % preserve in memory if desired
      if opt.wantmemoryoutputs(whmodel)==1
        results{whmodel} = struct;
        for p=1:length(allvars)
          results{whmodel}.(allvars{p}) = eval(allvars{p});
        end
      end

    end

    %%%%%

    function badness = calcbadness(xvals,validcolumns,stimix,results,sessionindicator)

    % function badness = calcbadness(xvals,validcolumns,stimix,results,sessionindicator)
    %
    % <xvals> is a cell vector of vectors of run indices
    % <validcolumns> is a cell vector, each element is the vector of trial indices associated with the run
    % <stimix> is a cell vector, each element is the vector of actual condition numbers occurring with a given run
    % <results> is a 1 x n with results. the first one is SPECIAL and is unregularized.
    % <sessionindicator> is 1 x RUNS with positive integers indicating run groupings for sessions.
    %   this is used only to perform the session-wise z-scoring for the purposes of hyperparameter evaluation.
    %
    % return <badness> as voxels x hyperparameters with the sum of the squared error from cross-validation.
    % the testing data consists of the beta weights from results(1), i.e. unregularized beta weights.
    % note that the squared error is expressed in the z-score units (given that we z-score the
    % single-trial beta weights prior to evaluation of the different hyperparameters).

    % note:
    % the unregularized betas set the stage for the session-wise normalization:
    % for each session, we determine a fixed mu and sigma that are applied to
    % the session under all of the various regularization levels.

    % initialize
    badness = zeros(size(results(1).modelmd{2},1),length(results));

    % calc
    alltheruns = 1:length(validcolumns);

    % z-score transform the single-trial beta weights
    for p=1:max(sessionindicator)
      wh = find(sessionindicator==p);
      whcol = catcell(2,validcolumns(wh));
      mn = mean(results(1).modelmd{2}(:,whcol),2);     % mean of unregularized case
      sd = std(results(1).modelmd{2}(:,whcol),[],2);   % std dev of unregularized case
      for q=1:length(results)
        results(q).modelmd{2}(:,whcol) = zerodiv(results(q).modelmd{2}(:,whcol) - repmat(mn,[1 length(whcol)]),repmat(sd,[1 length(whcol)]),0,0);
      end
    end

    % do cross-validation
    for xx=1:length(xvals)

      % calc
      testix = xvals{xx};                    % which runs are testing, e.g. [3 4]
      trainix = setdiff(alltheruns,testix);  % which runs are training, e.g. [1 2 5 6 7 8 9 10 11 12]
      
      % calc
      testcols = catcell(2,validcolumns(testix));    % vector of trial indices in the testing data
      traincols = catcell(2,validcolumns(trainix));  % vector of trial indices in the training data
      testids = catcell(2,stimix(testix));           % vector of condition-ids in the testing data
      trainids = catcell(2,stimix(trainix));         % vector of condition-ids in the training data
      
      % calculate cross-validation performance
      for ll=1:length(results)
    %    hashrec = cell(1,max(testids));  % speed-up by caching results
        for ttt=1:length(testids)
          haveix = find(trainids==testids(ttt));  % which training trials match the current condition-id?
          if ~isempty(haveix)
            
            % NOTE:
            % testcols(ttt) tells us which trial in the testing runs to pull betas for (these are 1-based trial numbers)
            % traincols(haveix) tells us the corresponding trials (isolated within the training runs) to pull betas for (these are 1-based trial numbers)

    %        if isempty(hashrec{testids(ttt)})
    %          hashrec{testids(ttt)} = mean(results(ll).modelmd{2}(:,traincols(haveix)),2);  % voxels x 1
    %          hashrec{testids(ttt)} = results(ll).modelmd{2}(:,traincols(haveix));  % voxels x instances
    %        end
            
            % compute squared error of all training betas against the current testing beta, and accumulate!!
            badness(:,ll) = badness(:,ll) + sum((results(ll).modelmd{2}(:,traincols(haveix)) - ...
                                                repmat(results(1).modelmd{2}(:,testcols(ttt)),[1 length(haveix)])).^2,2);  % NOTICE the use of results(1)

          end
        end
      end

    end

    %%%%%%%%%%%%%%%%%%% JUNK:

      % DEPRECATED
      %
      % % visualize
      % figureprep; hold on;
      % rvals = [1 3 5 10 20 30];
      % cmap0 = jet(length(rvals));
      % for pp=1:length(rvals)
      %   temp = glmbadness(onoffR2(:)>rvals(pp),:);
      %   plot(0:opt.numpcstotry,calczscore(median(temp,1)),'-','Color',cmap0(pp,:));
      % end
      % straightline(pcnum,'v','k-');
      % xlabel('number of pcs');
      % ylabel('median badness, z-scored');
      % figurewrite('checkbadness',[],[],outputtempdir);
      % 
      % % visualize  [PERHAPS GO BACK TO LINEAR; USE SCATTERSPARSE?]
      % rvals = [1 5 20];
      % colors = {'r' 'g' 'b'};
      % for p=1:opt.numpcstotry
      %   figureprep([100 100 900 900]);
      %   for cc=1:length(rvals)
      %     temp = glmbadness(onoffR2(:)>rvals(cc),:);
      %     scatter(log(temp(:,1)),log(temp(:,1+p)),[colors{cc} '.']);
      %   end
      %   axissquarify;
      %   %ax = axis;
      %   %axis([0 ax(2) 0 ax(2)]);
      %   xlabel('log error for no pcs');
      %   ylabel(sprintf('log error for %d pcs',p));
      %   figurewrite(sprintf('scatter%02d',p),[],[],outputtempdir);
      % end

