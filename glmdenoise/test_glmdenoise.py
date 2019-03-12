import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings
from itertools import compress
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
whitedesign = []
maxpolydeg = []
pmatrices = []
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
    pmatrices.append(pmatrix)
    polynomials.append(gpm.projectionmatrix(pmatrix))

    # append the whithened data
    whitedata.append(np.dot(polynomials[ii], y))

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
        design.append(X)
        whitedesign.append(np.dot(combinedmatrix[ii], X))


if opt['hrfmodel'] == 'optimise':
    hrfparams = ohrf.optimiseHRF(design,
                                 whitedata,
                                 TR,
                                 seedhrf,
                                 combinedmatrix)
    # update hrf and design
    hrf = hrfparams["hrf"]
    design = hrfparams["convdesign"]
    whitedesign = hrfparams["whitedesign"]
elif opt['hrfmodel'] == 'assume':
    # case assume. update hrfparams for book keeping
    hrfparams = dict()
    hrfparams["hrf"] = seedhrf
    hrfparams["convdesign"] = design
    hrfparams["whitedesign"] = whitedesign

# mean data and mask
mean_image = np.vstack(data).mean(0).reshape(*dims[1:])
mean_mask = mean_image > np.percentile(mean_image, 99) / 2

results = ohrf.crossval(whitedesign, whitedata, polynomials)

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
    u, s, vt = np.linalg.svd(white_pool * white_pool.T)  # * noise_pool.T)
    u = u[:, :20]
    u = u / np.std(u, 0)
    run_PCAs.append(u)

for n_pca in range(20):
    r2s = []
    modelfits = []
    for run in range(n_runs):
        # fit data using all the other runs
        mask = np.arange(n_runs) != run

        trainIndices = np.where(mask)[0]

        # whiten the design and data with pc regressors
        pccombinedmatrix = []
        whitetraindata = []
        whitetraindesign = []
        for ti in trainIndices:
            pctrain = run_PCAs[ti][:, :n_pca]
            pmat = pmatrices[ti]
            pmatcombined = gpm.projectionmatrix(
                np.c_[pmat, pctrain])
            pccombinedmatrix.append(pmatcombined)
            whitetraindata.append(np.dot(pmatcombined, data[ti]))
            whitetraindesign.append(np.dot(pmatcombined, design[ti]))

        stackdesign = np.vstack(whitetraindesign)

        betas = ohrf.mtimesStack(ohrf.olsmatrix(stackdesign), whitetraindata)

        # whiten the left out design
        whitetestdesign = np.dot(gpm.projectionmatrix(
            np.c_[pmatrices[run], run_PCAs[run][:, :n_pca]]), design[run])

        modelfit = np.dot(whitetestdesign, betas)
        modelfits.append(modelfit)

    # now we whiten the modelfits

    calccodStack(data[p], modelfitwhitemodelfit[p])

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

all_r2s = np.array(all_r2s)  # npc x voxel
best_mask = np.any(all_r2s > 0, 0) & mean_mask.flatten()
xval = np.median(all_r2s[:, best_mask], 1)
xval = np.median(all_r2s[:, all_r2s.mean(0) > 0], 1)
select_pca = select_noise_regressors(np.asarray(pca_r2))

plt.plot(pca_r2)
plt.plot(select_pca, pca_r2[select_pca], "o")
plt.show()


"""
dlist =
for dm in dms:

"""
