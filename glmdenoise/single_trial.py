import os
import numpy as np
import matplotlib.pyplot as plt
from glmdenoise.defaults import default_params
from glmdenoise.check_inputs import check_inputs
from glmdenoise.utils.normalisemax import normalisemax
from glmdenoise.utils.gethrf import getcanonicalhrf, getcanonicalhrflibrary
from glmdenoise.utils.chunking import chunking
from glmdenoise.utils.make_image_stack import make_image_stack
from glmdenoise.utils.findtailthreshold import findtailthreshold
from glmdenoise.utils.glm_estimatemodel import glm_estimatemodel
from ipdb import set_trace

dir0 = os.path.dirname(os.path.realpath(__file__))


class glm_estimatesingletrial():

    def __init__(self, params=None):
        """glm singletrial denoise constructor

        This function computes up to four model outputs (called type-A (ONOFF),
        type-B (FITHRF), type-C (FITHRF_GLMDENOISE), and type-D
        (FITHRF_GLMDENOISE_RR)),and either saves the model outputs to disk,
        or returns them in <results>, or both,depending on what the user
        specifies.

        There are a variety of cases that you can achieve. Here are some
        examples:

        - wantlibrary=1, wantglmdenoise=1, wantfracridge=1 [Default]
            A = simple ONOFF model
            B = single-trial estimates using a tailored HRF for every voxel
            C = like B but with GLMdenoise regressors added into the model
            D = like C but with ridge regression regularization (tailored to
                each voxel)

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
            Model type B is computed, but using least-squares-separate instead
            of OLS. Other model types, if computed, use OLS.

        Note that if you set wantglmdenoise=1, you MUST have repeats of
        conditions andan associated cross-validation scheme (<params.xvalscheme>),
        UNLESS you specify params.pcstop = -B. In other words, you can perform
        wantglmdenoise without any cross-validation, but you need to provide
        params.pcstop = -B.

        Note that if you set wantfracridge=1, you MUST have repeats of
        conditions and an associated cross-validation scheme
        (<params.xvalscheme>), UNLESS you specify a single scalar params.fracs.
        In other words, you can perform wantfracridge without any
        cross-validation, but you need to provide params.fracs as a scalar.

        Arguments:
        __________

        params (dict): Dictionary of parameters. Optional

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
            any integer between 0 and params.numpcstotry. Note that if -B case is
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
        """

        params = params or dict()
        for key, _ in default_params.items():
            params[key] = params.get(key) or default_params[key]

        self.params = params
        self.extra_regressors = params['extra_regressors']
        self.n_pcs = params['n_pcs']
        self.n_jobs = params['n_jobs']
        self.n_boots = params['n_boots']
        self.hrfparams = {}
        self.results = dict()

    def fit(self, design, data, stimdur, tr, outputdir=None):
        """
        Arguments:
        __________

        <design> is the experimental design. There are two possible cases:
        1. A where A is a matrix with dimensions time x conditions.
            Each column should be zeros except for ones indicating condition
            onsets.
        2. [A1, A2, ... An] where each of the A's are like the previous case.
            The different A's correspond to different runs, and different runs
            can have different numbers of time points. However, all A's must
            have the same number of conditions.
        Note that we ultimately compute single-trial response estimates (one
        estimate for each condition onset), and these will be provided in
        chronological order. However, by specifying that a given condition
        occurs more than one time over the course of the experiment, this
        information can and will be used for cross-validation purposes.

        <data> is the time-series data with dimensions X x Y x Z x time or a
         list vector of elements that are each X x Y x Z x time. XYZ can be
         collapsed such that the data are given as a 2D matrix (units x time),
         which is useful for surface-format data.
         The dimensions of <data> should mirror that of <design>. For example,
         <design> and <data> should have the same number of runs, the same
         number of time points, etc.
         <data> should not contain any NaNs. We automatically convert <data> to
         single format if not already in single format.
         <stimdur> is the duration of a trial in seconds. For example, 3.5
         means that you expect the neural activity from a given trial to last
         for 3.5 s.

        <tr> is the sampling rate in seconds. For example, 1 means that we get
         a new time point every 1 s. Note that <tr> applies to both <design>
         and <data>.

        <outputdir> (optional) is a directory to which files will be written.
         (If the directory does not exist, we create it; if the directory
         already exists, we delete its contents so we can start fresh.) If you
         set <outputdir> to None, we will not create a directory and no files
         will be written.
         Default is 'GLMestimatesingletrialoutputs' (created in the current
         working directory).


        Returns:
        __________

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

        <HRFindexrun> is HRFiniex separated by run

        <noisepool> indicates voxels selected for the noise pool

        <pcregressors> indicates the full set of candidate GLMdenoise
         regressors that were found

        <glmbadness> is cross-validation results for GLMdenoise

        <pcvoxels> is the set of voxels used to summarize GLMdenoise
         cross-validation results

        <xvaltrend> is the summary GLMdenoise cross-validation result on which
                    pcnum selection is done

        <pcnum> is the number of PCs that were selected for the final model

        <FRACvalue> is the fractional regularization level chosen for each
         voxel

        <scaleoffset> is the scale and offset applied to RR estimates to best
                    match the unregularized result

        History:
        [MATLAB]
        - 2020/08/22 - Implement params.sessionindicator. Also, now the
                        cross-validation units now reflect
                        the "session-wise z-scoring" hyperparameter selection
                        approach; thus, the cross-
                        validation units have been CHANGED relative to prior
                        analyses!
        - 2020/05/14 - Version 1.0 released!
                        (Tweak some documentation; output more results; fix a
                        small bug (params.fracs(1)~=1).)
        - 2020/05/12 - Add pcvoxels output.
        - 2020/05/12 - Initial version. Beta version. Use with caution.
        """

        # DEAL WITH INPUTS
        params = self.params

        # xyz can either be a tuple of dimensions x y z
        # or a boolean indicating that data was 2D
        data, design, xyz = check_inputs(data, design)

        # keep class bound data and design
        self.data = data
        self.design = design

        # calc
        numruns = len(design)

        if xyz:
            numvoxels = np.prod(xyz)
        else:
            numvoxels = self.data[0].shape[0]

        # check number of time points and truncate if necessary
        for p in np.arange(len(data)):
            if self.data[p].shape[1] > self.design[p].shape[0]:
                print(
                    f'WARNING: run {p} has more time points'
                    'in <data> than <design>. We are truncating'
                    '<data>.\n')
                self.data[p] = self.data[p][:, np.arange(self.design.shape[0])]

            if self.data[p].shape[1] < self.design[p].shape[0]:
                print(
                    f'WARNING: run {p} has more time points in <design>'
                    'than <data>. We are truncating <design>.\n')
                self.design[p] = self.design[p][np.arange(self.data[p].shape[0]), :]

        # inputs
        if 'xvalscheme' not in params:
            params['xvalscheme'] = np.arange(numruns)

        if 'sessionindicator' not in params:
            params['sessionindicator'] = np.ones((1, numruns))

        if 'maxpolydeg' not in params:
            params['maxpolydeg'] = [
                np.arange(
                    round(
                        ((self.data[r].shape[1]*tr)/60)/2) + 1
                    ) for r in np.arange(numruns)]

        if 'hrftoassume' not in params:
            params['hrftoassume'] = normalisemax(
                getcanonicalhrf(stimdur, tr),
                dim='global'
            )

        if 'hrflibrary' not in params:
            params['hrflibrary'] = getcanonicalhrflibrary(stimdur, tr).T

        # deal with length issues and other miscellaneous things
        if type(params['maxpolydeg']) is int:
            params['maxpolydeg'] = np.tile(
                params['maxpolydeg'], numruns
            ).tolist()

        # normalise maximal amplitude on hrfs
        params['hrftoassume'] = normalisemax(
            params['hrftoassume'],
            dim='global'
        )

        params['hrflibrary'] = normalisemax(params['hrflibrary'], 0)
        params['fracs'] = np.unique(params['fracs'])[::-1]
        np.testing.assert_equal(
            np.all(params['fracs'] > 0),
            True,
            err_msg='fracs must be greater than 0')

        np.testing.assert_equal(
            np.all(params['fracs'] <= 1),
            True,
            err_msg='fracs must be less than or equal to 1')

        if xyz and outputdir is not None:
            wantfig = 1  # if outputdir is not None, we want figures
        else:
            wantfig = 0

        # deal with output directory
        if outputdir is None:
            cwd = os.getcwd()
            outputdir = os.path.join(cwd, 'GLMestimatesingletrialoutputs')

        if os.path.exists(outputdir):
            import shutil
            shutil.rmtree(outputdir)
            os.makedirs(outputdir)
        else:
            os.makedirs(outputdir)

        if np.any(params['wantfileoutputs']):
            errm = 'specify an <outputdir> in order to get file outputs'
            np.testing.assert_equal(
                type(outputdir),
                str,
                err_msg=errm)

        # deal with special library stuff
        if params['wantlibrary'] == 0:
            params['hrflibrary'] = params['hrftoassume']

        # calc
        # if the data was passed as 3d, unpack xyz
        if xyz:
            nx, ny, nz = xyz

        nh = params['hrflibrary'].shape[1]

        # figure out chunking scheme
        chunks = chunking(
            np.arange(numvoxels),
            int(np.ceil(numvoxels/np.ceil(numvoxels/params['chunklen']))))

        # deal with special cases
        if params['wantglmdenoise'] == 1:
            errm = '<wantglmdenoise> is 1, but you didnt request type C nor D'
            test = np.any(
                    params['wantfileoutputs'][-2:]
                    ) or np.any(params['wantmemoryoutputs'][-2:])
            np.testing.assert_equal(
                test, True,
                err_msg=errm)

        if params['wantfracridge'] == 1:
            test = params['wantfileoutputs'][3] == 1 \
                or params['wantmemoryoutputs'][3] == 1
            np.testing.assert_equal(
                test, True,
                err_msg='<wantfracridge> is 1, but you did not request type D')

        if params['wantlss'] == 1:
            test = params['wantfileoutputs'][1] == 1 \
                    or params['wantmemoryoutputs'][1] == 1
            np.testing.assert_equal(
                test, True,
                err_msg='<wantlss> is 1, but you did not request type B')

        # initialize output
        results = []

        # PRE-PROCESSING FOR THE EXPERIMENTAL DESIGN

        # calculate the number of trials
        # number of trials in each run
        numtrialrun = np.asarray(
            [np.sum(x.flatten()) for x in self.design]).astype(int).tolist()
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
        for p in np.arange(len(self.design)):
            designSINGLE.append(np.zeros((self.design[p].shape[0], numtrials)))

            run_validcolumns = []
            # loop through the volumes for that run
            for q in np.arange(self.design[p].shape[0]):
                # if a condition was presented on that volume
                # find which
                temp = np.where(self.design[p][q, :])[0]

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
        design0 = [np.sum(x, axis=1)[:, np.newaxis] for x in self.design]
        optB = {
            'extra_regressors': params['extra_regressors'],
            'maxpolydeg': params['maxpolydeg'],
            'wantpercentbold': params['wantpercentbold'],
            'suppressoutput': 0
        }
        results0 = glm_estimatemodel(
            design0,
            self.data,
            stimdur,
            tr,
            'assume',
            params['hrftoassume'],
            0,
            optB
            )[0]

        onoffR2 = results0['R2']
        meanvol = results0['meanvol']

        # save to disk if desired
        if params['wantfileoutputs'][whmodel] == 1:
            file0 = os.path.join(outputdir, 'TYPEA_ONOFF.npy')
            print(f'*** Saving results to {file0}. ***\n')
            np.save(file0, onoffR2, meanvol, xyz)

        # figures
        if wantfig:
            """ TODO
            port normalizerange.m and add to makeimstack
            """
            plt.imshow(
                make_image_stack(onoffR2.reshape(xyz)),
                vmin=0,
                vmax=100,
                cmap='hot'
            )
            plt.colorbar()
            plt.savefig(os.path.join(outputdir, 'onoffR2.png'))
            plt.close('all')
            plt.imshow(make_image_stack(meanvol.reshape(xyz)), cmap='gray')
            plt.colorbar()
            plt.savefig(os.path.join(outputdir, 'meanvol.png'))
            plt.close('all')

        # preserve in memory if desired, and then clean up
        if params['wantmemoryoutputs'][whmodel] == 1:
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

        if 'brainR2' not in params:
            params['brainR2'] = thresh

        if 'pcR2cutoff' not in params:
            params['pcR2cutoff'] = thresh

        # FIT TYPE-B MODEL [FITHRF]

        # The approach:
        # (1) Fit single trials.
        # (2) Evaluate the library of HRFs (or the single assumed HRF).
        #     Choose based on R2 for each voxel.

        # if the user does not want file output nor memory output AND
        # if the number of HRFs to choose
        # from is just 1, we can short-circuit this whole endeavor!
        if params['wantfileoutputs'][1] == 0 and \
                params['wantmemoryoutputs'][1] == 0 and \
                params['hrflibrary'].shape[1] == 1:

            # short-circuit all of the work
            HRFindex = np.ones(numvoxels)  # easy peasy

        else:

            # define
            whmodel = 2

            # initialize
            FitHRFR2 = np.zeros(
                (numvoxels, nh),
                dtype=np.float32)
            # X x Y x Z x HRFs with R2 values (all runs)
            FitHRFR2run = np.zeros(
                (numvoxels, numruns, nh),
                dtype=np.float32)
            # X x Y x Z x runs x HRFs with R2 separated by runs
            modelmd = np.zeros(
                (numvoxels, numtrials),
                dtype=np.float32)
            # X x Y x Z x trialbetas
            optC = {
                    'extra_regressors': params['extra_regressors'],
                    'maxpolydeg': params['maxpolydeg'],
                    'wantpercentbold': params['wantpercentbold'],
                    'suppressoutput': 1
            }

            # loop over chunks
            print('*** FITTING TYPE-B MODEL (FITHRF) ***\n')
            for z in np.arange(len(chunks)):

                this_chunk = chunks[z]
                n_inchunk = len(this_chunk)

                print(f'working on chunk {z+1} of {len(chunks)}.\n')
                data_chunk = [datx[this_chunk, :] for datx in self.data]
                # do the fitting and accumulate all the betas
                modelmd0 = np.zeros(
                    (n_inchunk, numtrials, nh),
                    dtype=np.float32)
                # someXYZ x trialbetas x HRFs
                for p in np.arange(nh):
                    print(f'\t working on hrf {p}')
                    results0 = glm_estimatemodel(
                        designSINGLE,
                        data_chunk,
                        stimdur,
                        tr,
                        'assume',
                        params['hrflibrary'][:, p],
                        0,
                        optC
                    )[0]

                    FitHRFR2[this_chunk, p] = results0['R2']
                    FitHRFR2run[this_chunk, :, p] = np.stack(
                        results0['R2run']).T
                    modelmd0[:, :, p] = results0['betasmd']

                # keep only the betas we want
                # ii shape someXYZ
                ii = np.argmax(FitHRFR2[this_chunk, :], axis=1)

                # tile it as someXYZ x numtrials
                iiflat = np.tile(
                    ii[:, np.newaxis], numtrials).flatten()

                # someXYZ x numtrials x nh
                modelmd0 = np.reshape(
                    modelmd0, [n_inchunk*numtrials, -1])

                # XYZ by n_trials
                modelmd[this_chunk, :] = modelmd0[np.arange(
                    n_inchunk*numtrials), iiflat].reshape(n_inchunk, -1)

            R2 = np.max(FitHRFR2, axis=1)  # R2 is XYZ
            HRFindex = np.argmax(FitHRFR2, axis=1)  # HRFindex is XYZ

            # also, use R2 from each run to select best HRF
            HRFindexrun = np.argmax(FitHRFR2run, axis=2).flatten()

            FitHRFR2run = np.reshape(
                FitHRFR2run,
                (numvoxels*numruns, nh))

            # using each voxel's best HRF, what are the corresponding R2run
            # values?
            R2run = FitHRFR2run[np.arange(
                numvoxels*numruns),
                HRFindexrun].reshape([numvoxels, -1])

            # FIT TYPE-B MODEL (LSS) INTERLUDE BEGIN

            # if params.wantlss, we have to use the determined HRFindex and
            # re-fit the entire dataset using LSS estimation. this will
            # simply replace 'modelmd' with a new version.
            # so that's what we have to do here.

            if params['wantlss']:

                # initalize
                modelmd = np.zeros((numvoxels, numtrials), dtype=np.float32)
                # X*Y*Z x trialbetas  [the final beta estimates]

                # loop over chunks
                print(
                    '*** FITTING TYPE-B MODEL'
                    '(FITHRF but with LSS estimation) ***\n')

                for z in np.arange(len(chunks)):
                    print(f'working on chunk {z+1} of {len(chunks)}.\n')

                    this_chunk = chunks[z]
                    n_inchunk = len(this_chunk)

                    # loop over possible HRFs
                    for hh in np.arange(nh):

                        # figure out which voxels to process.
                        # this will be a vector of indices into the small
                        # chunk that we are processing.
                        # our goal is to fully process this set of voxels!
                        goodix = np.flatnonzero(
                            HRFindex[this_chunk] == hh)

                        data0 = \
                            [x[this_chunk, :][goodix, :] for x in self.data]

                        # calculate the corresponding indices relative to the
                        # full volume
                        temp = np.zeros(HRFindex.shape)
                        temp[this_chunk] = 1
                        relix = np.flatnonzero(temp)[goodix]

                        # define options
                        optA = {'extra_regressors': params['extra_regressors'],
                                'maxpolydeg': params['maxpolydeg'],
                                'wantpercentbold': params['wantpercentbold'],
                                'suppressoutput': 1
                                }

                        # do the GLM
                        cnt = 0
                        for rrr in np.arange(len(designSINGLE)):  # each run
                            for ccc in np.arange(numtrialrun[rrr]):  # each trial
                                designtemp = designSINGLE[rrr]
                                designtemp = np.c_[
                                    designtemp[:, cnt+ccc],
                                    np.sum(
                                        designtemp[:, np.setdiff1d(
                                            np.arange(
                                                designtemp.shape[1]
                                                ),
                                            cnt+ccc)],
                                        axis=1
                                    )
                                ]
                                results0, cache = glm_estimatemodel(
                                    designtemp,
                                    data0[rrr],
                                    stimdur,
                                    tr,
                                    'assume',
                                    params['hrflibrary'][:, hh],
                                    0,
                                    optA
                                )
                                modelmd[relix, cnt+ccc] = \
                                    results0['betasmd'][:, 0]

                            cnt = cnt + numtrialrun[rrr]

            # FIT TYPE-B MODEL (LSS) INTERLUDE END

            # save to disk if desired
            if params['wantfileoutputs'][whmodel] == 1:
                file0 = os.path.join(outputdir, 'TYPEB_FITHRF.npy')
                print(f'*** Saving results to {file0}. ***\n')
                np.save(
                    file0,
                    FitHRFR2,
                    FitHRFR2run,
                    HRFindex,
                    HRFindexrun,
                    R2,
                    R2run,
                    modelmd,
                    meanvol
                )

            # figures?
            if wantfig:
                """ TODO
                port normalizerange.m and add to makeimstack
                """
                plt.imshow(
                    make_image_stack(HRFindex.reshape(xyz)),
                    vmin=0,
                    vmax=nh)
                plt.colorbar()
                plt.savefig(os.path.join(outputdir, 'HRFindex.png'))
                plt.close('all')


            # preserve in memory if desired, and then clean up
            if params['wantmemoryoutputs'][whmodel] == 1:
                results['lss'] = {
                    'FitHRFR2': FitHRFR2,
                    'FitHRFR2run': FitHRFR2run,
                    'HRFindex': HRFindex,
                    'HRFindexrun': HRFindexrun,
                    'R2': R2,
                    'R2run': R2run,
                    'modelmd': modelmd,
                    'meanvol': meanvol
                }

