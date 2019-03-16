from numba import autojit, prange
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from glmdenoise.utils.make_design_matrix import make_design
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
        if extra_regressors and extra_regressors[0].any():
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
    for run in tqdm(range(n_runs), desc='Cross-validating run'):
        # fit data using all the other runs
        mask = np.arange(n_runs) != run

        betas = fit_runs(
            list(compress(whitened_data, mask)),
            list(compress(whitened_design, mask)))

        # predict left-out run with vanilla design matrix
        yhat = design[run] @ betas

        y = data[run]
        # get polynomials
        polynomials = make_poly_matrix(y.shape[0], poly_degs)
        # project out polynomials from data and prediction
        y = make_project_matrix(polynomials) @ y
        yhat = make_project_matrix(polynomials) @ yhat

        nom, denom = R2_nom_denom(y, yhat)
        nom_denom.append((nom, denom))

    # calculate R2
    nom = np.array(nom_denom).sum(0)[0, :]
    denom = np.array(nom_denom).sum(0)[1, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        r2s = np.nan_to_num(1 - (nom / denom)) * 100
    return r2s


class GLMdenoise():
    """
    Bad noise, bad noise,
    Whatcha gonna do
    Whatcha gonna do
    When it comes for you
    """

    def __init__(self, design, data, tr=2.0, n_jobs=10, n_pcs=20, n_boots=100):
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
        self.tr = tr
        self.n_pcs = n_pcs
        self.dims = data[0].shape
        self.n_jobs = n_jobs
        self.n_boots = n_boots
        self.n_runs = len(data)
        self.xval = []
        self.PCA_R2s = []
        # calculate polynomial degrees
        max_poly_deg = int(((data[0].shape[0] * tr) / 60) // 2) + 1
        self.poly_degs = np.arange(max_poly_deg)

        self.mean_image = np.vstack(data).mean(0)
        self.mean_mask = self.mean_image > np.percentile(
            self.mean_image, 99) / 2

        # reduce data
        self.data = [d[:, self.mean_mask].astype(
            np.float16) for d in self.data]

    def fit(self):
        """
        """
        print('Making initial fit')
        r2s_initial = cross_validate(
            self.data, self.design, poly_degs=self.poly_degs)
        print('Done!')

        print('Getting noise pool')
        self.noise_pool_mask = (r2s_initial < 0)

        print('Calculating PCs...')
        run_PCAs = []
        for run in self.data:
            noise_pool = run[:, self.noise_pool_mask]

            polynomials = make_poly_matrix(noise_pool.shape[0], self.poly_degs)
            noise_pool = make_project_matrix(polynomials) @ noise_pool

            noise_pool = normalize(noise_pool, axis=0)

            noise_pool = noise_pool @ noise_pool.T
            u = np.linalg.svd(noise_pool)[0]
            u = u[:, :21]
            u = u / np.std(u, 0)
            run_PCAs.append(u)

        self.pc_regressors = []
        for n_pc in range(self.n_pcs):
            self.pc_regressors.append([pc[:, :n_pc] for pc in run_PCAs])
        print('Done!')

        print('Fit data with PCs...')
        self.PCA_R2s = Parallel(n_jobs=self.n_jobs, backend='threading')(
            delayed(cross_validate)(self.data, self.design,
                                    self.pc_regressors[x], self.poly_degs) for x in tqdm(
                range(self.n_pcs), desc='Number of PCs'))
        print('Done!')

        # calculate best number of PCs
        self.PCA_R2s = np.asarray(self.PCA_R2s)
        best_mask = np.any(self.PCA_R2s > 0, 0)
        self.xval = np.nanmedian(self.PCA_R2s[:, best_mask], 1)
        self.select_pca = select_noise_regressors(np.asarray(self.xval))
        print(f'Selected {self.select_pca} number of PCs')

        whitened_data, whitened_design = whiten_data(
            self.data, self.design,
            self.pc_regressors[self.select_pca],
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
        print('Done!')

        print('Calculating standard error and final fit')
        boot_betas = np.array(boot_betas)
        self.final_fit = np.median(boot_betas, 0)
        n_conds = boot_betas.shape[1]
        self.standard_error = np.zeros((self.final_fit.shape))
        for cond in range(n_conds):
            percentiles = np.percentile(
                boot_betas[:, cond, :], [16, 84], axis=0)
            self.standard_error[cond, :] = (
                percentiles[1, :] - percentiles[0, :])/2

        self.poolse = np.sqrt(np.mean(self.standard_error**2, axis=0))
        with np.errstate(divide="ignore", invalid="ignore"):
            self.pseudo_t_stats = np.apply_along_axis(
                lambda x: x/self.poolse, 1, self.final_fit)
        print('Done')

    def plot_figures(self):
        plt.plot(self.xval)
        plt.plot(self.select_pca, self.xval[self.select_pca], "o")
        plt.show()
