# temp = diag(1./len)*((X'*X)\X');
from sklearn.preprocessing import normalize
import numpy as np

def make_project_matrix(X):
    #X = normalize(X)
    X = np.mat(X)
    return np.eye(X.shape[0]) - (X*(np.linalg.inv(X.T*X)*X.T))

def get_poly_matrix(n, degrees):
    """
    n:  number of points
    degrees: vector of polynomial degrees
    """
    time_points = np.linspace(-1, 1, n)[np.newaxis].T
    polys = np.zeros((n, len(degrees)))

    # Loop over degrees
    for i, d in enumerate(degrees):
        polyvector = np.mat(time_points**d)

        if i > 0: # project out the other polynomials
            polyvector = make_project_matrix(polys[:, :i]) * polyvector

        polys[:, i] = normalize(polyvector.T)
    return polys # make_project_matrix(polys)
