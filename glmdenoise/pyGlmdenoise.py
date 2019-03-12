import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from nistats.design_matrix import make_design_matrix
from nistats.reporting import plot_design_matrix
from itertools import compress
import statsmodels.api as sm
from tqdm import tqdm
import warnings
from utils.get_poly_matrix import *
import numpy
from scipy.sparse.linalg import svd

warnings.simplefilter(action="ignore", category=FutureWarning)


warnings.simplefilter(action="ignore", category=FutureWarning)


def rsquared(y, yhat):
    with np.errstate(divide="ignore", invalid="ignore"):
        nom = np.sum((y - yhat) ** 2, axis=0)
        denom = np.sum((y - y.mean(axis=0)) ** 2, axis=0)  # correct denominator
        denom = np.sum(y ** 2, axis=0)  # Kendricks denominator
        rsq = np.nan_to_num(1 - nom / denom)

    # remove inf-values because we might have voxels outside the brain
    rsq[rsq < -1] = -1
    return rsq

def fit_separate_runs(runs, DM, extra_regressors = False):
    #run_lens = [r.shape[0] for r in runs]
    #constants = get_constants(run_lens)
    if extra_regressors and extra_regressors[0].any():

        for i, (y, X) in enumerate(zip(runs, DM)):
            polynomials = get_poly_matrix(X.shape[0], [0, 1, 2, 3, 4])
            X = np.c_[X, extra_regressors[i]]
            runs[i] = make_project_matrix(polynomials) * y
            DM[i] = make_project_matrix(polynomials) * X
        # n_regs = extra_regressors[0].shape[1]
        # betas = np.zeros((DM[0].shape[1]+n_regs, runs[0].shape[1]))
        X = np.vstack(DM)
        y = np.vstack(runs)
        # X = np.c_[X, constants]

        # regs = np.vstack(extra_regressors)
        # X = np.c_[X, regs]

        betas = sm.OLS(y, X).fit().params

        # for run, X, reg in zip(runs,DM, extra_regressors):
        #     X = np.c_[X, reg]
        #     polynomials = get_poly_matrix(X.shape[0], [0, 1, 2, 3, 4])
        #     run = make_project_matrix(polynomials) * run
        #     X = make_project_matrix(polynomials) * X
        #     betas += sm.OLS(run, X).fit().params
    else:
        for i, (y, X) in enumerate(zip(runs, DM)):
            polynomials = get_poly_matrix(X.shape[0], [0, 1, 2, 3, 4])
            runs[i] = make_project_matrix(polynomials) * y
            DM[i] = make_project_matrix(polynomials) * X
        betas = np.zeros((DM[0].shape[1], runs[0].shape[1]))

        # n_regs = extra_regressors[0].shape[1]
        # betas = np.zeros((DM[0].shape[1]+n_regs, runs[0].shape[1]))
        X = np.vstack(DM)
        # X = np.c_[X, constants]
        y = np.vstack(runs)

        betas = sm.OLS(y, X).fit().params

        # for run, X in zip(runs,DM):
        #     polynomials = get_poly_matrix(X.shape[0], [0, 1, 2, 3, 4])
        #     run = np.mat(make_project_matrix(polynomials) * run)
        #     X = np.mat(make_project_matrix(polynomials) * X)
        #     betas += sm.OLS(run, X).fit().params
    return betas


def get_constants(run_lens):
    """ Calculates a sparse array with 1s describing a run on its corresponding
        column

    Args:
        run_lens (int): data

    Returns:
        array of size sum(run_lens) x len(run_lens)
    """
    tot_len = np.sum(run_lens)
    const = np.zeros((tot_len, len(run_lens)))

    for i, rlen in enumerate(run_lens):
        const[i * rlen : (i + 1) * rlen, i] = 1
    return const


def R2_nom_denom(y, yhat):
    """ Calculates the nominator and denomitor for calculating R-squared

    Args:
        y (array): data
        yhat (array): predicted data data

    Returns:
        nominator (float or array), denominator (float or array)
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        nom = np.sum((y - yhat) ** 2, axis=0)
        denom = np.sum(y ** 2, axis=0)  # Kendricks denominator
    return nom, denom


data = []
design = []
n_runs = 6
for ii in range(n_runs):
    y = np.load(f"data/sub_sub-01_slice_15_run_{ii+1:02d}.npy")
    y = np.moveaxis(y, 2,0)
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

    X = make_design_matrix(
        frame_times, events, hrf_model="glover", drift_model=None
    )

    data.append(y)
    design.append(X.values[:, :-1])

# kendricks formula for the number of degrees to use
maxpolydeg = int(((data[0].shape[0] * TR) / 60) // 2)
# mean data and mask
mean_image = np.vstack(data).mean(0).reshape(*dims[1:])
mean_mask = mean_image > np.percentile(mean_image, 99) / 2

"""
Get initial fit to select noise pool
"""
preds = []
nom_denom = []
r2s = []
for run in range(n_runs):
    # fit data using all the other runs
    mask = np.arange(n_runs) != run
    betas = fit_separate_runs(
        list(compress(data, mask)), list(compress(design, mask))
    )

    # left out data
    # project out polynomials
    polynomials = get_poly_matrix(design[run].shape[0], [0, 1, 2, 3, 4])
    this_design = make_project_matrix(polynomials) * design[run]
    yhat = np.mat(this_design.dot(betas))

    y = np.mat(data[run])
    # project out polynomials
    y = make_project_matrix(polynomials) * y

    nom, denom = R2_nom_denom(np.array(y), np.array(yhat))
    r2s.append(rsquared(np.array(y), np.array(yhat)))
    preds.append(np.array(yhat))
    nom_denom.append((nom, denom))

"""
Calculate R2s
"""
with np.errstate(divide="ignore", invalid="ignore"):
    nom = np.array(nom_denom).sum(0)[0, :]
    denom = np.array(nom_denom).sum(0)[1, :]
    r2s_vanilla = np.nan_to_num(1 - nom / denom)*100

"""
Plot R-squares
"""
# r2s_vanilla = np.vstack(r2s).mean(0)
fig, ax = plt.subplots(2, 2, figsize=(8, 6))
ax = ax.flatten()

sns.distplot(r2s_vanilla, color="green", ax=ax[0])

ax[0].set_title('Python R2 distribution')

# plt_img[plt_img < 0.2] = 0
sns.heatmap(
    r2s_vanilla.reshape((80, 80), order='F'),
    #mask=~mean_mask,
    ax=ax[1],
    xticklabels=False,
    yticklabels=False,
    square=True,
)

ax[1].set_title('Python R2 brain')
matlab_r2 = loadmat('../r2.mat')['pcR2']

sns.distplot(np.nan_to_num(matlab_r2.flatten()), color="green", ax=ax[2])
ax[2].set_title('Matlab R2 distribution')

# plt_img[plt_img < 0.2] = 0
sns.heatmap(
    np.nan_to_num(matlab_r2),
    #mask=~mean_mask,
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

sns.heatmap(noise_pool_mask.reshape(*dims[1:]).astype(int),xticklabels=False, yticklabels=False)
sns.heatmap(best_vox.reshape(*dims[1:]).astype(int),xticklabels=False, yticklabels=False)
"""
mask_flat = mean_mask.reshape(-1, order='F')
# noise_pool_mask = (r2s_vanilla < 0) & mask
noise_pool_mask = (r2s_vanilla < 0) & mask_flat
best_vox = (
    r2s_vanilla > 0
)  # (r2s_vanilla > np.percentile(r2s_vanilla,95)) & mask

matlab_noise = loadmat('../noise.mat')['noisepool'].astype(bool)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
sns.heatmap(
    r2s_vanilla.reshape(*dims[1:], order='F'),
    mask=~noise_pool_mask.reshape(*dims[1:]),
    xticklabels=False,
    yticklabels=False,
    ax=ax[0],
)
sns.heatmap(
    r2s_vanilla.reshape(*dims[1:], order='F'),
    mask=~matlab_noise,
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
matlab_pca = loadmat('../first_pca.mat')['a']

# split PCAs
run_PCAs = []
for run in data:
    noise_pool = run[:, matlab_noise.flatten(order='F')]

    polynomials = get_poly_matrix(noise_pool.shape[0], [0, 1, 2, 3, 4])
    noise_pool = make_project_matrix(polynomials) * np.mat(noise_pool)

    noise_pool = np.mat(normalize(noise_pool,axis=0))

    u, s, Vh = svd(noise_pool * noise_pool.T)#, lapack_driver='gesvd')
    u, s, vt = np.linalg.svd(noise_pool* noise_pool.T)# * noise_pool.T)
    u =  u[:,:20]
    u = u / np.std(u, 0)
    run_PCAs.append(u)

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
diff= matlab_pca-run_PCAs[0]
sns.heatmap(
    diff,
    mask=np.abs(diff) >0.01,
    xticklabels=False,
    yticklabels=False,
    ax=ax[2],
)
ax[0].set_title("Numpy PCA")
ax[1].set_title("Matlab PCA")
ax[2].set_title("Difference")
plt.show()


pca_r2 = []
mean_r2 = []
all_r2s =[]
for n_pca in tqdm(range(20)):
    nom_denom = []
    r2s = []
    for run in range(n_runs):
        # fit data using all the other runs
        mask = np.arange(n_runs) != run

        # tmpy = np.vstack(compress(data, mask))
        # X = np.vstack((compress(design, mask)))

        # pcas = np.vstack((compress(run_PCAs, mask)))
        # X = np.c_[X, pcas[:, :n_pca]]


        pc_regressors = [pc[:, :n_pca] for pc in compress(run_PCAs, mask)]
        betas = fit_separate_runs(
            list(compress(data, mask)),
            list(compress(design, mask)),
            pc_regressors,
        )

        # left out data
        # project out polynomials
        polynomials = get_poly_matrix(design[run].shape[0], [0, 1, 2, 3, 4])
        pcas = run_PCAs[run]
        left_out_X = np.c_[design[run], pcas[:, :n_pca]]
        left_out_X = make_project_matrix(polynomials) * left_out_X
        yhat = left_out_X.dot(betas)

        # project out polynomials
        y = make_project_matrix(polynomials) * data[run]
        y, yhat = np.asarray(y), np.asarray(yhat)

        nom, denom = R2_nom_denom(y, yhat)
        nom_denom.append((nom, denom))
        r2s.append(rsquared(y, yhat))

    mean_r2.append(np.median(np.vstack(r2s).mean(0)[best_vox]))
    with np.errstate(divide="ignore", invalid="ignore"):
        nom = np.array(nom_denom).sum(0)[0, :]
        denom = np.array(nom_denom).sum(0)[1, :]
        r2s = np.nan_to_num(1 - nom / denom)
    all_r2s.append(r2s)
    pca_r2.append(np.median(r2s[best_vox]))

all_r2s = np.array(all_r2s) # npc x voxel
best_mask = np.any(all_r2s > 0, 0) & mean_mask.flatten()
xval  = np.median(all_r2s[:, best_mask], 1)
xval = np.median(all_r2s[:, all_r2s.mean(0)>0],1)
select_pca = select_noise_regressors(np.asarray(pca_r2))

plt.plot(pca_r2)
plt.plot(select_pca, pca_r2[select_pca], "o")
plt.show()


"""
dlist =
for dm in dms:

"""
