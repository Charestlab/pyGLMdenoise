import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings
from utils import get_poly_matrix as gpm
from utils import optimiseHRF as ohrf
from utils.getcanonicalhrf import getcanonicalhrf
from utils.normalisemax import normalisemax
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize

# import scipy.io as sio
warnings.simplefilter(action="ignore", category=FutureWarning)


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
        const[i * rlen: (i + 1) * rlen, i] = 1
    return const


# options
opt = {}
opt['hrfmodel'] = 'optimise'
opt['extraregressors'] = None


data = []
whitedata = []
design = []
maxpolydeg = []
polynomials = []
combinedmatrix = []
n_runs = 8
stimdur = 0.5
TR = 0.764

# initialise hrf
seedhrf = normalisemax(getcanonicalhrf(stimdur, TR))

for ii in range(n_runs):
    # load the data and append
    y = np.load(f"data/sub_sub-01_slice_15_run_{ii+1:02d}.npy")
    y = np.swapaxes(y, 0, 2)
    dims = y.shape
    y = y.reshape([y.shape[0], -1])

    # append the non-whitened data
    data.append(y)

    # get n volumes
    n_vols = y.shape[0]

    # get polynomials
    maxpolydeg.append(int(((n_vols * TR) / 60) // 2)+1)
    pmatrix = gpm.constructpolynomialmatrix(
        n_vols, list(range(maxpolydeg[ii])))
    polynomials.append(gpm.projectionmatrix(pmatrix))

    # append the whithened data
    whitedata.append(polynomials[ii] * y)

    # handle extra regressors
    if opt['extraregressors'] is not None:
        pass
        # exmatrix.append(gpm.projectionmatrix(opt['extraregressors'][ii]))
        # combinedmatrix.append(gpm.projectionmatrix(
        #         np.concatenate(pmatrix, opt['extraregressors'][ii], axis=1)))
    else:
        combinedmatrix.append(polynomials[ii])

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

    if opt['hrfmodel'] == 'optimise':
        # if optimise hrf, we pass the design as a stack of events pandas
        # this is because we need the event deltas design matrix
        design.append(events)
    else:
        X = ohrf.make_design(events, TR, n_vols, seedhrf)
        design.append(np.dot(combinedmatrix[ii], X))


if opt['hrfmodel'] == 'optimise':
    hrfparams = ohrf.optimiseHRF(design,
                                 whitedata,
                                 TR,
                                 seedhrf,
                                 combinedmatrix)
    # update hrf and design
    hrf = hrfparams["hrf"]
    design = hrfparams["convdesign"]
elif opt['hrfmodel'] == 'assume':
    # case assume. update hrfparams for book keeping
    hrfparams = dict()
    hrfparams["hrf"] = seedhrf
    hrfparams["convdesign"] = design

# mean data and mask
mean_image = np.vstack(data).mean(0).reshape(*dims[1:])
mean_mask = mean_image > np.percentile(mean_image, 99) / 2

results = ohrf.crossval(design, whitedata, polynomials)

# check with a figure
fig, ax = plt.subplots(1, 2, figsize=(8, 3))
sns.distplot(results['R2'], color="green", ax=ax[0])

plt_img = results['R2'].copy()
# plt_img[plt_img < 0.2] = 0
sns.heatmap(
    plt_img.reshape((80, 80)),
    # mask=~mean_mask,
    cmap="hot",
    ax=ax[1],
    xticklabels=False,
    yticklabels=False,
    square=True,
)
plt.show()

"""
plots noise pool

sns.heatmap(noise_pool_mask.reshape(
    *dims[1:]).astype(int),xticklabels=False, yticklabels=False)
sns.heatmap(best_vox.reshape(
    *dims[1:]).astype(int),xticklabels=False, yticklabels=False)
"""
mask = mean_mask.reshape(-1)
# noise_pool_mask = (r2s_vanilla < 0) & mask
noise_pool_mask = (results['R2'] < np.percentile(
    results['R2'][mask], 5)) & mask
noise_pool_mask = (results['R2'] < 0) & (
    mean_image.reshape(-1)
    > np.percentile(mean_image[mean_mask].flatten(), 99) / 2
)

# show the noisepool
volpool = np.zeros((6400))
volpool[noise_pool_mask] = 1
sns.heatmap(volpool.reshape((80, 80)))
plt.show()

best_vox = (
    results['R2'] > 0
)  # (r2s_vanilla > np.percentile(r2s_vanilla,95)) & mask

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
sns.heatmap(
    results['R2'].reshape(*dims[1:]),
    mask=~noise_pool_mask.reshape(*dims[1:]),
    xticklabels=False,
    yticklabels=False,
    ax=ax[0],
)
sns.heatmap(
    results['R2'].reshape(*dims[1:]),
    mask=~best_vox.reshape(*dims[1:]),
    xticklabels=False,
    yticklabels=False,
    ax=ax[1],
)
ax[0].set_title("Noise pool")
ax[1].set_title("best voxels")
plt.show()

"""
Loop over number of PCAs
calculate cross-validated fit
"""

# split PCAs
run_PCAs = []
for q, run in enumerate(data):
    noise_pool = run[:, noise_pool_mask]
    white_pool = np.dot(polynomials[q], noise_pool)
    white_pool = normalize(white_pool, axis=1)
    # polynomials = get_poly_matrix(noise_pool.shape[0], [0, 1, 2, 3, 4])
    # noise_pool = make_project_matrix(polynomials) * noise_pool
    u, s, vt = svds(np.mat(white_pool) * np.mat(white_pool.T), k=20)
    u = u / np.std(u, 0)
    run_PCAs.append(u)

pca_r2 = []
mean_r2 = []
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

        # n_regressors = design[run].shape[1] + n_pca
        # betas = sm.OLS(tmpy, X).fit().params[:n_regressors,:]

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
    pca_r2.append(np.median(r2s[best_vox]))
select_pca = select_noise_regressors(np.asarray(pca_r2))

plt.plot(pca_r2)
plt.plot(select_pca, pca_r2[select_pca], "o")
plt.show()


"""
dlist =
for dm in dms:

"""
