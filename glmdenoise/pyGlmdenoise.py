from ipdb import set_trace
from numba import autojit, prange
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from glmdenoise.utils.make_design_matrix import make_design
from glmdenoise.utils.optimiseHRF import *
from glmdenoise.report import Report
from itertools import compress
from scipy.io import loadmat
from tqdm import tqdm
import warnings
from joblib import Parallel, delayed
from glmdenoise.utils.make_poly_matrix import *
warnings.simplefilter(action="ignore", category=FutureWarning)


def R2_nom_denom(y, yhat):
    """ Calculates the nominator and denomitor for calculating R-squared

    Args:
        y (array): data
        yhat (array): predicted data data

    Returns:
        nominator (float or array), denominator (float or array)
    """
    y, yhat = np.array(y), np.array(yhat)
    with np.errstate(divide="ignore", invalid="ignore"):
        nom = np.sum((y - yhat) ** 2, axis=0)
        denom = np.sum(y ** 2, axis=0)  # Kendricks denominator
    return nom, denom


def whiten_data(data, design, extra_regressors=False, poly_degs=np.arange(5)):
    """[summary]

    Arguments:
        data {[type]} -- [description]
        design {[type]} -- [description]

    Keyword Arguments:
        extra_regressors {bool} -- [description] (default: {False})
        poly_degs {[type]} -- [description] (default: {np.arange(5)})

    Returns:
        [type] -- [description]
    """

    # whiten data
    whitened_data = []
    whitened_design = []

    for i, (y, X) in enumerate(zip(data, design)):
        polynomials = make_poly_matrix(X.shape[0], poly_degs)
        if extra_regressors:
            if extra_regressors[i].any():
                polynomials = np.c_[polynomials, extra_regressors[i]]

        whitened_design.append(make_project_matrix(polynomials) @ X)
        whitened_data.append(make_project_matrix(polynomials) @ y)

    return whitened_data, whitened_design


@autojit
def fit_runs(data, design):
    """Fits a least square of combined runs.
       The matrix addition is equivalent to concatenating the list of data and the list of
       design and fit it all at once. However, this is more memory efficient.
    Arguments:
        runs {list} -- List of runs. Each run is an TR x voxel sized array
        DM {list} -- List of design matrices. Each design matrix
                     is an TR x predictor sizec array

    Returns:
        [array] -- betas from fit
    """

    X = np.vstack(design)
    X = np.linalg.inv(X.T @ X) @ X.T

    betas = 0
    start_col = 0

    for run in prange(len(data)):
        n_vols = data[run].shape[0]
        these_cols = np.arange(n_vols) + start_col
        betas += X[:, these_cols] @ data[run]
        start_col += data[run].shape[0]

    return betas


def cross_validate(data, design, extra_regressors=False, poly_degs=np.arange(5)):
    """[summary]

    Arguments:
        data {[type]} -- [description]
        design {[type]} -- [description]

    Keyword Arguments:
        extra_regressors {bool} -- [description] (default: {False})
        poly_degs {[type]} -- [description] (default: {np.arange(5)})

    Returns:
        [type] -- [description]
    """

    whitened_data, whitened_design = whiten_data(
        data, design, extra_regressors, poly_degs)

    n_runs = len(data)
    nom_denom = []
    betas = []
    r2_runs = []
    for run in tqdm(range(n_runs), desc='Cross-validating run'):
        # fit data using all the other runs
        mask = np.arange(n_runs) != run

        betas.append(fit_runs(
            list(compress(whitened_data, mask)),
            list(compress(whitened_design, mask))))

        # predict left-out run with vanilla design matrix
        yhat = design[run] @ betas[run]

        y = data[run]
        # get polynomials
        polynomials = make_poly_matrix(y.shape[0], poly_degs)
        # project out polynomials from data and prediction
        y = make_project_matrix(polynomials) @ y
        yhat = make_project_matrix(polynomials) @ yhat

        # get run-wise r2s
        nom, denom = R2_nom_denom(y, yhat)
        with np.errstate(divide="ignore", invalid="ignore"):
            r2_runs.append(np.nan_to_num(1 - (nom / denom)))

        nom_denom.append((nom, denom))

    # calculate global R2
    nom = np.array(nom_denom).sum(0)[0, :]
    denom = np.array(nom_denom).sum(0)[1, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        r2s = np.nan_to_num(1 - (nom / denom))
    return (r2s, r2_runs, betas)


class GLMdenoise():
    """
    Bad noise, bad noise,
    Whatcha gonna do
    Whatcha gonna do
    When it comes for you
    """

    def __init__(self, design, data, params, n_jobs=10, n_pcs=11, n_boots=10):
        """[summary]

        Arguments:
            design {[type]} -- [description]
            data {[type]} -- [description]

        Keyword Arguments:
            tr {float} -- TR in seconds (default: {2})
            n_jobs {int} -- [description] (default: {10})
            n_pcs {int} -- [description] (default: {20})
            n_boots {int} -- [description] (default: {100})
        """

        self.design = design
        self.data = data
        self.params = params
        self.params['n_pcs'] = n_pcs
        self.tr = params['tr']
        self.extra_regressors = params['extra_regressors']
        self.n_pcs = n_pcs
        self.dims = data[0].shape
        self.n_jobs = n_jobs
        self.n_boots = n_boots
        self.n_runs = len(data)
        self.hrfparams = {}
        self.results = dict()

        # calculate polynomial degrees
        max_poly_deg = int(((data[0].shape[0] * self.tr) / 60) // 2) + 1
        self.poly_degs = np.arange(max_poly_deg)

        self.results['mean_image'] = np.vstack(data).mean(0)
        self.results['mean_mask'] = self.results['mean_image'] > np.percentile(
            self.results['mean_image'], 99) / 2

        # reduce data
        self.data = [d[:, self.results['mean_mask']].astype(
            np.float16) for d in self.data]

    def fit(self):
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
        set_trace()
        best_mask = np.any(
            self.results['PCA_R2s'] > self.params['R2thresh'], 0)
        self.results['xval'] = np.nanmedian(
            self.results['PCA_R2s'][:, best_mask], 1)
        select_pca = select_noise_regressors(
            np.asarray(self.results['xval']))
        self.results['select_pca'] = select_pca
        print(f'Selected {select_pca} number of PCs')

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

        stackdesign = np.vstack(whitened_design)
        modelfits = mtimesStack(olsmatrix(stackdesign), whitened_data)
        self.results['R2s'] = calccodStack(
            whitened_data,
            modelfits)
        """ TO DO
        self.results['R2runs'] = [calccod(whitened_data[c_run], modelfits[c_run], 0)
                                  for c_run in range(self.n_runs)]
        """
        print('Done')

    def plot_figures(self):
        # start a new report with figures
        report = Report()
        report.spatialdims = self.params['xyzsize']

        # plot solutions
        title = 'HRF fit'
        report.plot_hrf(self.hrfparams['hrfseed'],
                        self.hrfparams['hrf'], self.tr, title)
        report.plot_image(self.hrfparams['hrffitvoxels'], title)

        for c_run in range(n_runs):
            report.plot_scatter_sparse(
                [
                    (self.results['PCA_R2s'][0],
                        self.results['PCA_R2s'][c_run]),
                    (self.results['PCA_R2s'][0][self.pcvoxels],
                        self.results['PCA_R2s'][c_run][self.pcvoxels]),
                ],
                xlabel='Cross-validated R^2 (0 PCs)',
                ylabel='Cross-validated R^2 ({p} PCs)',
                title='PCscatter{p}',
                crosshairs=True,
            )

        # plot voxels for noise regressor selection
        title = 'Noise regressor selection'
        report.plot_noise_regressors_cutoff(self.results['xval'], self.n_pcs,
                                            title='Chosen number of regressors')
        report.plot_image(self.results['noise_pool_mask'], title)

        # various images
        report.plot_image(self.results['mean_image'], 'Mean volume')
        report.plot_image(self.results['noise_pool_mask'], 'Noise Pool')
        report.plot_image(self.results['mean_mask'], 'Noise Exclude')
        if self.hrfparams['hrffitmask'] != 1:
            report.plot_image(self.hrfparams['hrffitmask'], 'HRFfitmask')
        if self.params['pcR2cutoffmask'] != 1:
            report.plot_image(self.params['pcR2cutoffmask'], 'PCmask')

        for pc in range(self.n_pcs):
            report.plot_image(
                self.results['PCA_R2s'][pc],
                'PCcrossvalidation%02d', dtype='range')
            report.plot_image(
                self.results['PCA_R2s'][pc],
                'PCcrossvalidationscaled%02d', dtype='scaled')

        report.plot_image(self.results['R2s'], 'FinalModel')
        """ TODO
        for r in range(n_runs):
            report.plot_image(
                self.results['R2srun'][r], 'FinalModel_run%02d')
        """
        # PC weights
        thresh = np.percentile(
            np.abs(np.vstack(self.results['PCA_weights']).ravel()), 99)
        for c_run in range(1, self.n_runs):
            for pc in range(1, self.n_pcas):
                report.plot_image(
                    self.results['PCA_weights'][pc][c_run].mean(axis=0),
                    'PCmap_run%02d_num%02d.png',
                    dtype='custom',
                    drange=[-thresh, thresh]
                )

        # stores html report
        report.save()

        print('Done')
