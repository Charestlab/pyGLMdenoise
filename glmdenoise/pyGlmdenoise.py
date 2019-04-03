import numpy as np
import numpy
from tqdm import tqdm
import warnings
from joblib import Parallel, delayed
from sklearn.preprocessing import normalize
from glmdenoise.utils.make_design_matrix import make_design
from glmdenoise.utils.optimiseHRF import mtimesStack, olsmatrix, optimiseHRF
from glmdenoise.select_noise_regressors import select_noise_regressors
from glmdenoise.utils.normalisemax import normalisemax
from glmdenoise.utils.gethrf import getcanonicalhrf
from glmdenoise.report import Report
from glmdenoise.defaults import default_params
from glmdenoise.whiten_data import whiten_data
from glmdenoise.fit_runs import fit_runs
from glmdenoise.cross_validate import cross_validate
from glmdenoise.utils.make_poly_matrix import make_poly_matrix, make_project_matrix
warnings.simplefilter(action="ignore", category=FutureWarning)


class GLMdenoise():
    """
    Bad noise, bad noise,
    Whatcha gonna do
    Whatcha gonna do
    When it comes for you
    """

    def __init__(self, params=None):
        """GlmDenoise constructor

        Arguments:
            params (dict): Dictionary of parameters. Optional
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

    def fit(self, design, data, tr):

        self.design = design
        self.tr = tr
        self.data = data
        self.n_runs = len(data)

        stimdur = numpy.median(design[0].duration.values)
        self.params['hrf'] = normalisemax(getcanonicalhrf(stimdur, tr))
        self.params['tr'] = tr

        # calculate polynomial degrees
        max_poly_deg = int(((data[0].shape[0] * self.tr) / 60) // 2) + 1
        self.poly_degs = np.arange(max_poly_deg)

        self.results['mean_image'] = np.vstack(data).mean(0)
        self.results['mean_mask'] = self.results['mean_image'] > np.percentile(
            self.results['mean_image'], 99) / 2

        # reduce data
        self.data = [d[:, self.results['mean_mask']].astype(
            np.float16) for d in self.data]


        """
        """
        if self.params['hrfmodel'] == 'optimise':
            print('Optimising HRF...')
            convdes = []
            polymatrix = []
            for run in range(self.n_runs):
                n_times = self.data[run].shape[0]
                X = make_design(self.design[run], self.tr,
                                n_times, self.params['hrf'])

                convdes.append(X)
                max_poly_deg = np.arange(
                    int(((X.shape[0] * self.tr) / 60) // 2) + 1)
                polynomials = make_poly_matrix(X.shape[0], max_poly_deg)
                if self.extra_regressors:
                    polynomials = np.c_[polynomials,
                                        self.extra_regressors[run]]
                polymatrix.append(make_project_matrix(polynomials))

            # optimise hrf requires whitening of the data
            whitened_data, whitened_design = whiten_data(
                self.data, convdes, self.extra_regressors, poly_degs=self.poly_degs)

            hrfparams = optimiseHRF(
                self.design,
                whitened_data,
                self.tr,
                self.params['hrf'],
                polymatrix)
            whitened_design = None
            whitened_data = None
            self.hrfparams = hrfparams
            self.design = self.hrfparams['convdesign']
        else:
            print('Assuming HRF...')
            convdes = []

            for run in range(self.n_runs):
                n_times = self.data[run].shape[0]
                X = make_design(self.design[run], self.tr,
                                n_times, self.params['hrf'])
                convdes.append(X)

            self.design = convdes
            self.hrfparams["hrf"] = self.params['hrf']
            self.hrfparams["hrffitvoxels"] = None

        print('Making initial fit')
        r2s_initial = cross_validate(
            self.data, self.design, self.extra_regressors, poly_degs=self.poly_degs)[0]
        print('Done!')

        print('Getting noise pool')
        self.results['noise_pool_mask'] = (
            r2s_initial < self.params['R2thresh'])

        print('Calculating PCs...')
        run_PCAs = []
        for run in self.data:
            noise_pool = run[:, self.results['noise_pool_mask']]

            polynomials = make_poly_matrix(noise_pool.shape[0], self.poly_degs)
            noise_pool = make_project_matrix(polynomials) @ noise_pool

            noise_pool = normalize(noise_pool, axis=0)

            noise_pool = noise_pool @ noise_pool.T
            u = np.linalg.svd(noise_pool)[0]
            u = u[:, :self.n_pcs+1]
            u = u / np.std(u, 0)
            run_PCAs.append(u)

        self.results['pc_regressors'] = []
        for n_pc in range(self.n_pcs):
            self.results['pc_regressors'].append(
                [pc[:, :n_pc] for pc in run_PCAs])
        print('Done!')

        print('Fit data with PCs...')
        PCresults = Parallel(
            n_jobs=self.n_jobs, backend='threading')(
            delayed(cross_validate)(self.data, self.design,
                                    self.results['pc_regressors'][x],
                                    self.poly_degs) for x in tqdm(
                range(self.n_pcs), desc='Number of PCs'))
        print('Done!')
        # calculate best number of PCs
        self.results['PCA_R2s'] = np.vstack(
            np.asarray(np.asarray(PCresults)[:, 0]))
        self.results['PCA_R2_runs'] = np.asarray(np.asarray(PCresults)[:, 1])
        self.results['PCA_weights'] = np.asarray(np.asarray(PCresults)[:, 2])
        self.best_mask = np.any(
            self.results['PCA_R2s'] > self.params['R2thresh'], 0)
        self.results['xval'] = np.nanmedian(
            self.results['PCA_R2s'][:, self.best_mask], 1)
        select_pca = select_noise_regressors(
            np.asarray(self.results['xval']))
        self.results['select_pca'] = select_pca
        print('Selected {} number of PCs'.format(select_pca))

        print('Bootstrapping betas (No Denoising).')
        n_runs = len(self.data)

        whitened_data, whitened_design = whiten_data(
            self.data, self.design,
            poly_degs=self.poly_degs)

        boot_betas = []
        boot_data = []
        boot_design = []
        for _ in range(self.n_boots):
            boot_inds = np.random.choice(np.arange(n_runs), n_runs)
            boot_data.append([self.data[ind] for ind in boot_inds])
            boot_design.append(
                [whitened_design[ind] for ind in boot_inds])

        boot_betas = Parallel(n_jobs=self.n_jobs, backend='threading')(
            delayed(fit_runs)(
                boot_data[x], boot_design[x]) for x in tqdm(
                range(self.n_boots), desc='Bootstrapping'))

        boot_betas = np.array(boot_betas)
        self.results['vanilla_fit'] = np.median(boot_betas, 0)
        print('Done!')

        print('Bootstrapping betas (Denoising).')
        whitened_data, whitened_design = whiten_data(
            self.data, self.design,
            self.results['pc_regressors'][self.results['select_pca']],
            self.poly_degs)

        print('Bootstrapping betas...')
        n_runs = len(self.data)
        boot_betas = []

        boot_data = []
        boot_design = []
        for _ in range(self.n_boots):
            boot_inds = np.random.choice(np.arange(n_runs), n_runs)
            boot_data.append([whitened_data[ind] for ind in boot_inds])
            boot_design.append([whitened_design[ind] for ind in boot_inds])

        boot_betas = Parallel(n_jobs=self.n_jobs, backend='threading')(
            delayed(fit_runs)(
                boot_data[x], boot_design[x]) for x in tqdm(
                range(self.n_boots), desc='Bootstrapping'))

        print('Calculating standard error and final fit')
        boot_betas = np.array(boot_betas)
        self.results['final_fit'] = np.median(boot_betas, 0)
        n_conds = boot_betas.shape[1]
        self.results['standard_error'] = np.zeros(
            (self.results['final_fit'].shape))
        for cond in range(n_conds):
            percentiles = np.percentile(
                boot_betas[:, cond, :], [16, 84], axis=0)
            self.results['standard_error'][cond, :] = (
                percentiles[1, :] - percentiles[0, :])/2

        self.results['poolse'] = np.sqrt(
            np.mean(self.results['standard_error']**2, axis=0))
        with np.errstate(divide="ignore", invalid="ignore"):
            self.results['pseudo_t_stats'] = np.apply_along_axis(
                lambda x: x/self.results['poolse'], 1, self.results['final_fit'])

        print('Calculating overall R2 of final fit...')

        # The below does not currently work, see #47
        # stackdesign = np.vstack(whitened_design)
        # modelfits = mtimesStack(olsmatrix(stackdesign), whitened_data)
        # self.results['R2s'] = calccodStack(whitened_data, modelfits)
        # self.results['R2runs'] = [calccod(whitened_data[c_run], modelfits[c_run], 0)
        #                           for c_run in range(self.n_runs)]
        print('Done')

    def plot_figures(self, report=None, spatialdims=None):

        # start a new report with figures
        if report is None:
            report = Report()
            report.spatialdims = self.params.get('xyzsize') or spatialdims

        if self.params.get('hrfmodel') == 'optimize':
            title = 'HRF fit'
            report.plot_hrf(
                self.hrfparams['seedhrf'],
                self.hrfparams['hrf'],
                self.tr,
                title
            )
            report.plot_image(self.hrfparams['hrffitvoxels'], title)

        for pc in range(self.n_pcs):
            report.plot_scatter_sparse(
                [
                    (self.results['PCA_R2s'][0],
                        self.results['PCA_R2s'][pc]),
                    (self.results['PCA_R2s'][0][self.best_mask],
                        self.results['PCA_R2s'][pc][self.best_mask]),
                ],
                xlabel='Cross-validated R^2 (0 PCs)',
                ylabel='Cross-validated R^2 ({} PCs)'.format(pc),
                title='PCscatter{}'.format(pc),
                crosshairs=True,
            )

        # plot voxels for noise regressor selection
        title = 'Noise regressor selection'
        report.plot_noise_regressors_cutoff(self.results['xval'],
                                            self.results['select_pca'],
                                            title='Chosen number of regressors')

        # various images
        report.plot_image(self.results['mean_image'], 'Mean volume')
        report.plot_image(self.results['noise_pool_mask'], 'Noise Pool')
        report.plot_image(self.results['mean_mask'], 'Noise Exclude')
        if self.hrfparams.get('hrffitmask', 1) != 1:
            report.plot_image(self.hrfparams['hrffitmask'], 'HRFfitmask')
        if self.params.get('pcR2cutoffmask', 1) != 1:
            report.plot_image(self.params['pcR2cutoffmask'], 'PCmask')

        for pc in range(self.n_pcs):
            report.plot_image(
                self.results['PCA_R2s'][pc],
                'PCcrossvalidation{}'.format(pc), dtype='range')
            report.plot_image(
                self.results['PCA_R2s'][pc],
                'PCcrossvalidationscaled{}'.format(pc), dtype='scaled')

        #report.plot_image(self.results['R2s'], 'FinalModel')
        """ TODO
        for r in range(n_runs):
            report.plot_image(
                self.results['R2srun'][r], 'FinalModel_run%02d')
        """
        # PC weights
        thresh = np.percentile(
            np.abs(np.vstack(self.results['PCA_weights']).ravel()), 99)
        for c_run in range(1, self.n_runs):
            for pc in range(1, self.n_pcs):
                # matrix to array for reshaping:
                weights = np.asarray(self.results['PCA_weights'][pc][c_run])
                report.plot_image(
                    weights.mean(axis=0),
                    'PCmap_run{}_num{}'.format(c_run, pc),
                    dtype='custom',
                    drange=[-thresh, thresh]
                )

        # stores html report
        report.save()

        print('Done')
