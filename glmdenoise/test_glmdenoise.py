import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from random import choices
import warnings
from itertools import compress
from utils import get_poly_matrix as gpm
from utils import optimiseHRF as ohrf
from utils.getcanonicalhrf import getcanonicalhrf
from utils.normalisemax import normalisemax
from scipy.linalg import svd
from sklearn.preprocessing import normalize
from select_noise_regressors import select_noise_regressors
import time
start = time.time()

# import scipy.io as sio
warnings.simplefilter(action="ignore", category=FutureWarning)

# options
opt = {}
opt['hrfmodel'] = 'optimise'
opt['extraregressors'] = None
opt['pcR2cutoff'] = 0
opt['pcR2cutoffmask'] = 1
opt['numboots'] = 100

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
    whitedata.append(polynomials[ii] @ y)

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
        whitedesign.append(combinedmatrix[ii] @ X)


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
# TO DO : check whether here we use data or whitedata
mean_image = np.vstack(data).mean(0).reshape(*dims[1:])
mean_mask = mean_image > np.percentile(mean_image, 99) / 2

results = ohrf.crossval(whitedesign, whitedata,
                        design, whitedata, polynomials)


"""
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
noise_pool_mask = (results['R2'] < opt['pcR2cutoff']) & (
    mean_image.reshape(-1)
    > np.percentile(mean_image[mean_mask].flatten(), 99) / 2
)

"""
# show the noisepool
volpool = np.zeros((6400))
volpool[noise_pool_mask] = 1
sns.heatmap(volpool.reshape((80, 80)))
plt.show()

best_vox = (
    results['R2'] > opt['pcR2cutoff']
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

"""
Loop over number of PCAs
calculate cross-validated fit
"""

# split PCAs
run_PCAs = []
for q, run in enumerate(data):
    noise_pool = run[:, noise_pool_mask]
    white_pool = polynomials[q] @ noise_pool
    white_pool = np.mat(normalize(white_pool, axis=0))
    u, s, Vh = svd(white_pool @ white_pool.T, lapack_driver='gesvd')
    # u, s, vt = np.linalg.svd(white_pool * white_pool.T)  # * noise_pool.T)
    u = u[:, :21]
    u = u / np.std(u, 0)
    run_PCAs.append(u)

pcR2 = []
for n_pca in range(21):
    print('cross-validating model with {} PCs...'.format(n_pca))
    pccombinedmatrix = []
    whitepcdata = []
    whitepcdesign = []
    for run in range(n_runs):
        # collect all the pcs and the projected data
        pctrain = run_PCAs[run][:, :n_pca]
        pmat = pmatrices[run]
        pmatcombined = gpm.projectionmatrix(
            np.c_[pmat, pctrain])
        pccombinedmatrix.append(pmatcombined)
        whitepcdata.append(pmatcombined @ data[run])
        whitepcdesign.append(pmatcombined @ design[run])

    pcresults = ohrf.crossval(
        whitepcdesign, whitepcdata, design, whitedata, polynomials)
    pcR2.append(pcresults["R2"])

# vertical stack of the PCR2
pcR2stack = np.vstack(pcR2)

# find the voxels that break the cutoff
pcvoxels = np.any(pcR2stack > opt['pcR2cutoff'], axis=0)

xval = np.nanmedian(pcR2stack[:, pcvoxels], 1)
select_pca = select_noise_regressors(np.asarray(xval))

"""
plt.plot(xval)
plt.plot(select_pca, xval[select_pca], "o")
plt.show()
"""
# bootstrap runs for vanilla fit
# using whitedesign, whitedata
bootbetas = []
for bootn in range(opt['numboots']):
    print(f'working on boot {bootn}')
    bootis = choices(range(n_runs), k=n_runs)
    bootdes = [whitedesign[x] for x in bootis]
    bootdat = [whitedata[x] for x in bootis]
    stackdesign = np.vstack(bootdes)
    bootbetas.append(np.asarray(ohrf.mtimesStack(
        ohrf.olsmatrix(stackdesign), bootdat)))
bootbetas = np.stack(bootbetas, axis=-1)

nvox, nconditions, nboot = bootbetas.shape
vanillafits = np.zeros((nvox, nconditions))
vanillase = np.zeros((nvox, nconditions))
for p in range(nconditions):
    temp = np.percentile(bootbetas[:, p, :], [16, 50, 84], axis=1).T
    vanillafits[:, p] = temp[:, 1]
    vanillase[:, p] = (temp[:, 2] - temp[:, 0])/2


# now do the final fit with the selected number of pcs
# here we bootstrap as well

n_pca = select_pca
whitepcdata = []
whitepcdesign = []
for run in range(n_runs):
    # collect all the pcs and the projected data
    pctrain = run_PCAs[run][:, :n_pca]
    pmat = pmatrices[run]
    pmatcombined = gpm.projectionmatrix(
        np.c_[pmat, pctrain])
    pccombinedmatrix.append(pmatcombined)
    whitepcdata.append(pmatcombined @ data[run])
    whitepcdesign.append(pmatcombined @ design[run])

bootdenoisebetas = []
for bootn in range(opt['numboots']):
    print(f'working on boot {bootn}')
    bootis = choices(range(n_runs), k=n_runs)
    bootdes = [whitepcdesign[x] for x in bootis]
    bootdat = [whitepcdata[x] for x in bootis]
    stackdesign = np.vstack(bootdes)
    bootdenoisebetas.append(np.asarray(ohrf.mtimesStack(
        ohrf.olsmatrix(stackdesign), bootdat)).T)

bootdenoisebetas = np.stack(bootdenoisebetas, axis=-1)

nvox, nconditions, nboot = bootdenoisebetas.shape
finalfits = np.zeros((nvox, nconditions))
modelse = np.zeros((nvox, nconditions))
for p in range(nconditions):
    temp = np.percentile(bootdenoisebetas[:, p, :], [16, 50, 84], axis=1).T
    finalfits[:, p] = temp[:, 1]
    modelse[:, p] = (temp[:, 2] - temp[:, 0])/2

poolse = np.sqrt(np.mean(modelse**2, axis=1))
pseudots = finalfits / modelse

# run function
print(f'it took {time.time()-start} seconds')

"""
dlist =
for dm in dms:

"""
