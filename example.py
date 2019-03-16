import time
from glmdenoise.utils.make_image_stack import make_image_stack
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from glmdenoise.utils.make_design_matrix import make_design
from glmdenoise.utils.gethrf import getcanonicalhrf
from glmdenoise.utils.normalisemax import normalisemax
from glmdenoise.utils.optimiseHRF import optimiseHRF

from glmdenoise.utils.make_poly_matrix import *
from glmdenoise import pyGlmdenoise as PYG
import warnings
import os
import glob
import nibabel as nib
from itertools import compress
warnings.simplefilter(action="ignore", category=FutureWarning)


def format_time(time):
    return f'{int(time//60)} minutes and {time-(time//60)*60:.2f} seconds'


stimdur = 0.5
TR = 0.764


"""
Load data
"""

fmri_folder = '/home/adf/charesti/Documents/sub-01'

runs = glob.glob(os.path.join(fmri_folder, 'ses*', 'func', '*preproc*nii.gz'))
runs.sort()
eventfs = glob.glob(os.path.join(fmri_folder, 'ses*', 'func', '*_events.tsv'))
eventfs.sort()
runs = compress(runs, np.arange(len(runs)) != 1)
eventfs = compress(eventfs, np.arange(len(eventfs)) != 1)

data = []
design = []
eventdesign = []
polymatrix = []
hrf = normalisemax(getcanonicalhrf(stimdur, TR))
for i, (run, event) in enumerate(zip(runs, eventfs)):
    print(f'run {i}')
    y = nib.load(run).get_data().astype(np.float32)
    dims = y.shape
    y = np.moveaxis(y, -1, 0)
    y = y.reshape([y.shape[0], -1])

    n_volumes = y.shape[0]

    # Load onsets and item presented
    onsets = pd.read_csv(event, sep='\t')["onset"].values
    items = pd.read_csv(event, sep='\t')["item"].values
    n_events = len(onsets)

    # Create design matrix
    events = pd.DataFrame()
    events["duration"] = [stimdur] * n_events
    events["onset"] = onsets
    events["trial_type"] = items

    eventdesign.append(events)
    X = make_design(events, TR, n_volumes)

    max_poly_deg = np.arange(int(((X.shape[0] * TR) / 60) // 2) + 1)
    polynomials = make_poly_matrix(X.shape[0], max_poly_deg)
    polymatrix.append(make_project_matrix(polynomials))
    data.append(polymatrix[i] @ y)

hrfparams = optimiseHRF(
    eventdesign,
    data,
    TR,
    hrf,
    polymatrix)

design = hrfparams['convdesign']

runs = glob.glob(os.path.join(fmri_folder, 'ses*', 'func', '*preproc*nii.gz'))
runs.sort()
eventfs = glob.glob(os.path.join(fmri_folder, 'ses*', 'func', '*_events.tsv'))
eventfs.sort()
runs = compress(runs, np.arange(len(runs)) != 1)
eventfs = compress(eventfs, np.arange(len(eventfs)) != 1)

data = []
for i, (run, event) in enumerate(zip(runs, eventfs)):
    print(f'run {i}')
    y = nib.load(run).get_data()
    dims = y.shape
    y = np.moveaxis(y, -1, 0)
    y = y.reshape([y.shape[0], -1])
    n_volumes = y.shape[0]
    data.append(y)

gd = PYG.GLMdenoise(design, data, tr=0.764, n_jobs=2)
start = time.time()
gd.fit()
print(f'Fit took {format_time(time.time()-start)}!')

# plot pseudo T statistics
brain = np.zeros(len(gd.mean_mask))
brain[gd.mean_mask] = gd.pseudo_t_stats.mean(0)
brain = brain.reshape(*dims[:-1])

stack = make_image_stack(brain)

plt.imshow(stack)
plt.show()
