import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from make_design_matrix import make_design
from itertools import compress
from scipy.io import loadmat
from tqdm import tqdm
import warnings
from utils.get_poly_matrix import *
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


def fit_runs(runs, DM, extra_regressors=False, poly_degs=np.arange(5)):
    """Fits a least square of combined runs by first whitening the data and design. 
       The matrix addition is equivalent to concatenating the list of data and the list of
       design and fit it all at once. However, this is more memory efficient. 
    Arguments:
        runs {list} -- List of runs. Each run is an TR x voxel sized array
        DM {list} -- List of design matrices. Each design matrix 
                     is an TR x predictor sizec array
    
    Keyword Arguments:
        extra_regressors {list} -- list of same length as runs and DM (default: {False})
        poly_degs {array} -- array of polynomial degrees to project 
                             from data and design (default: {np.arange(5)})
    
    Returns:
        [array] -- betas from fit
    """

    # whiten data
    for i, (y, X) in enumerate(zip(runs, DM)):
        polynomials = get_poly_matrix(X.shape[0], poly_degs)
        if extra_regressors and extra_regressors[0].any():
            polynomials = np.c_[polynomials, extra_regressors[i]]
        DM[i] = make_project_matrix(polynomials) @ X
        runs[i] = make_project_matrix(polynomials) @ y
    
    X = np.vstack(DM)
    X = np.linalg.inv(X.T @ X) @ X.T

    betas = 0
    start_col = 0
    for run in (runs):
        n_vols = run.shape[0]
        these_cols = np.arange(n_vols) + start_col
        betas += X[:, these_cols] @ run
        start_col += run.shape[0]
    return betas

def cross_validate(data, design, extra_regressors=False):
    """[summary]
    
    Arguments:
        data {list} -- [description]
        design {list} -- [description]
    
    Keyword Arguments:
        extra_regressors {bool/list} -- [description] (default: {False})
    
    Returns:
        [array] -- [description]
    """

    nom_denom = []
    for run in tqdm(range(n_runs), desc='Cross-validating run'):
        # fit data using all the other runs
        mask = np.arange(n_runs) != run
        if extra_regressors and extra_regressors[0].any():
            betas = fit_runs(
                list(compress(data, mask)),
                list(compress(design, mask)),
                list(compress(extra_regressors, mask)))
        else:
            betas = fit_runs(
                list(compress(data, mask)),
                list(compress(design, mask)))

        # predict left-out run with vanilla design matrix
        yhat = design[run] @ betas

        y = data[run]
        # get polynomials
        polynomials = get_poly_matrix(y.shape[0], poly_degs)
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

data = []
design = []
n_runs = 6
hrf = np.load('hrf.npy')    
for ii in range(n_runs):
    y = np.load(f"data/sub_sub-01_slice_15_run_{ii+1:02d}.npy")
    #y = np.moveaxis(y, 2,0)
    y = np.swapaxes(y, 0, 2)
    dims = y.shape
    y = y.reshape([y.shape[0], -1])

    stimdur = 0.5
    TR = 0.764
    n_scans = y.shape[0]
    frame_times = np.arange(n_scans) * TR

    # Load onsets and item presented
    onsets = pd.read_csv(f"data/sub_sub-01_slice_15_run_{ii+1:02d}.csv")[
        "onset"
    ].values
    items = pd.read_csv(f"data/sub_sub-01_slice_15_run_{ii+1:02d}.csv")[
        "item"
    ].values
    n_events = len(onsets)

    # Create design matrix
    events = pd.DataFrame()
    events["duration"] = [stimdur] * n_events
    events["onset"] = onsets
    events["trial_type"] = items

    X = make_design(events, TR, n_scans, hrf)
    data.append(y)
    design.append(X)

# kendricks formula for the number of degrees to use
max_poly_deg = int(((data[0].shape[0] * TR) / 60) // 2) + 1
poly_degs = np.arange(max_poly_deg)

# mean data and mask
mean_image = np.vstack(data).mean(0).reshape(*dims[1:])
mean_mask = mean_image > np.percentile(mean_image, 99) / 2

"""
Get initial fit to select noise pool
"""

r2s_vanilla = cross_validate(data, design)

"""
Plot R-squares
"""
fig, ax = plt.subplots(2, 2, figsize=(8, 6))
ax = ax.flatten()

sns.distplot(r2s_vanilla, color="green", ax=ax[0]) 
ax[0].set_xlim([-10, 20])
ax[0].set_title('Python R2 distribution')

sns.heatmap(
    r2s_vanilla.reshape((80, 80)),
    ax=ax[1],
    xticklabels=False,
    yticklabels=False,
    square=True,
)

ax[1].set_title('Python R2 brain')
matlab_r2 = loadmat('../r2.mat')['pcR2']

sns.distplot(np.nan_to_num(matlab_r2.flatten()), color="green", ax=ax[2])
ax[2].set_title('Matlab R2 distribution')
ax[2].set_xlim([-10, 20])

sns.heatmap(
    np.nan_to_num(matlab_r2),
    ax=ax[3],
    xticklabels=False,
    yticklabels=False,
    square=True,
)
ax[3].set_title('Matlab R2 brain')
plt.tight_layout()
plt.show()

"""
plots noise pool
"""
mask_flat = mean_mask.reshape(-1)
noise_pool_mask = (r2s_vanilla < 0) & mask_flat

matlab_noise = loadmat('../noise.mat')['noisepool'].astype(bool)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
sns.heatmap(
    noise_pool_mask.reshape(*dims[1:]).astype(int),
    xticklabels=False,
    yticklabels=False,
    ax=ax[0],
)
sns.heatmap(
    matlab_noise.astype(int),
    xticklabels=False,
    yticklabels=False,
    ax=ax[1],
)
ax[0].set_title("Numpy noise pool")
ax[1].set_title("Matlab noise pool")
plt.show()

"""
Loop over number of PCAs
calculate cross-validated fit
"""

# split PCAs
# Daniel PCA
mat_regs = loadmat('../pc_regressors.mat')['pcregressors'][0]
run_PCAs = mat_regs
run_PCAs = []
for run in data:
    noise_pool = run[:, noise_pool_mask]

    polynomials = get_poly_matrix(noise_pool.shape[0], poly_degs)
    noise_pool = make_project_matrix(polynomials) * np.mat(noise_pool)

    noise_pool = normalize(noise_pool, axis=0)

    noise_pool = noise_pool @ noise_pool.T
    u, s, vt = np.linalg.svd(noise_pool)
    u =  u[:, :21]
    u = u / np.std(u, 0)
    run_PCAs.append(u)

"""
plot difference to matlabs first PCA 
"""
matlab_pca = loadmat('../first_pca.mat')['a']
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
sns.heatmap(
    run_PCAs[0],
    xticklabels=False,
    yticklabels=False,
    ax=ax[0],
)
sns.heatmap(
    matlab_pca,
    xticklabels=False,
    yticklabels=False,
    ax=ax[1],
)
diff= run_PCAs[0]-matlab_pca
sns.heatmap(
    diff,
    xticklabels=False,
    yticklabels=False,
    ax=ax[2],
)
ax[0].set_title("Matlab PCA")
ax[1].set_title("Daniel PCA")
ax[2].set_title("Difference")
plt.show()
all_r2s = []
for n_pca in tqdm(range(21), desc='Number of PCs'):
    pc_regressors = [pc[:, :n_pca] for pc in run_PCAs]
    all_r2s.append(cross_validate(data, design, pc_regressors))
all_r2s = np.array(all_r2s)
all_r2s =[]
for n_pca in tqdm(range(21)):
    nom_denom = []
    r2s = []
    for run in range(n_runs):
        # fit data using all the other runs
        mask = np.arange(n_runs) != run

        pc_regressors = [pc[:, :n_pca] for pc in compress(run_PCAs, mask)]
        betas = fit_runs(
            list(compress(data, mask)),
            list(compress(design, mask)),
            pc_regressors,
        )

        # left out data
        this_data = data[run]
        this_design = design[run]

        # predict
        yhat = this_design @ betas

        # get polynomials
        polynomials = get_poly_matrix(this_design.shape[0], poly_degs)
        # project out polynomials
        y = make_project_matrix(polynomials) @ this_data
        yhat = make_project_matrix(polynomials) @ yhat

        # calculate nominator and denominator for R2
        nom, denom = R2_nom_denom(y, yhat)
        nom_denom.append((nom, denom))

    with np.errstate(divide="ignore", invalid="ignore"):
        nom = np.array(nom_denom).sum(0)[0, :]
        denom = np.array(nom_denom).sum(0)[1, :]
        r2s = np.nan_to_num(1 - (nom / denom)) * 100
    all_r2s.append(r2s)


all_r2s = np.array(all_r2s) # npc x voxel
best_mask = np.any(all_r2s > 0, 0) & mask_flat
xval  = np.nanmedian(all_r2s[:, best_mask], 1)
select_pca = select_noise_regressors(np.asarray(xval))

plt.plot(xval)
plt.plot(select_pca, xval[select_pca], "o")
plt.show()


mat_r2 = loadmat('../pca_r2.mat')['pca_r2']
best_mask = np.any(mat_r2 > 0, 0)
xval  = np.nanmedian(mat_r2[:, best_mask], 1)
select_pca = select_noise_regressors(np.asarray(xval))

plt.plot(xval)
plt.plot(select_pca, xval[select_pca], "o")
plt.show()
