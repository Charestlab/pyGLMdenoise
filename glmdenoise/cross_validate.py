import numpy as np
from itertools import compress
from tqdm import tqdm
import warnings
from glmdenoise.utils.make_poly_matrix import make_poly_matrix, make_project_matrix
from glmdenoise.whiten_data import whiten_data
from glmdenoise.r2_nom_denom import R2_nom_denom
from glmdenoise.fit_runs import fit_runs
warnings.simplefilter(action="ignore", category=FutureWarning)


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
