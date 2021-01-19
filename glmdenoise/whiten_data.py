
import numpy as np
from glmdenoise.utils.make_poly_matrix import (make_poly_matrix,
                                               make_project_matrix)
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)


def whiten_data(data, design, extra_regressors=False, poly_degs=None):
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
    if poly_degs is None:
        poly_degs = np.arange(5)

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
