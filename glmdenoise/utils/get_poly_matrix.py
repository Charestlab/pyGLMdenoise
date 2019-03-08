from sklearn.preprocessing import normalize
import numpy as np

def make_project_matrix(X):
    """ Calculates a projection matrix

    Args:
        X (array): design matrix

    Returns:
        array: Projection matrix size of X.shape[0] x X.shape[0]
    """
    X = np.mat(X)
    return np.eye(X.shape[0]) - (X*(np.linalg.inv(X.T*X)*X.T))

def get_poly_matrix(n, degrees):
    """Calculates a matrix of polynomials used to regress them out of your data

    Args:
        n (int): number of points
        degrees (array): vector of polynomial degrees

    Returns:
        array: array of n x len(degrees)
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



def select_noise_regressors(r2_nrs, pcstop=1.05):
    """How many components to include

    Args:
        r2_nrs (ndarray): Model fit value per solution
        pcstop (float, optional): Defaults to 1.05.

    Returns:
        int: Number of noise regressors to include
    """
    numpcstotry = r2_nrs.size - 1

    # this is the performance curve that starts at 0 (corresponding to 0 PCs)
    curve = r2_nrs - r2_nrs[0]

    # initialize (this will hold the best performance observed thus far)
    best = -np.Inf
    for p in range(1, numpcstotry):

      # if better than best so far
      if curve[p] > best:

        # record this number of PCs as the best
        chosen = p
        best = curve[p]

        # if we are within opt.pcstop of the max, then we stop.
        if (best * pcstop) >= curve.max():
            break

    return chosen
