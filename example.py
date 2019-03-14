import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from glmdenoise.utils.make_design_matrix import make_design
from glmdenoise import pyGlmdenoise as PYG
import warnings
import os, glob
import nibabel as nib
from itertools import compress
warnings.simplefilter(action="ignore", category=FutureWarning)

"""
Load data
"""

fmri_folder = '/home/dlindh/AMS_AB_RSA/preproc/fmriprep/sub-01'


runs = glob.glob(os.path.join(fmri_folder, 'ses*', 'func', '*preproc*nii.gz'))
runs.sort()
events = glob.glob(os.path.join(fmri_folder, 'ses*', 'func', '*_events.tsv'))
events.sort()

runs = compress(runs, np.arange(len(runs)) != 1)
events = compress(events, np.arange(len(events)) != 1)

data = []
design = []
hrf = np.load('hrf.npy')    
for i, (run, event) in enumerate(zip(runs, events)):
    print(f'run {i}')
    y = nib.load(run).get_data()
    dims = y.shape
    y = np.moveaxis(y, -1,0)
    y = y.reshape([y.shape[0], -1], order='F')

    stimdur = 0.5
    TR = 0.764
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

    X = make_design(events, TR, n_volumes, hrf)
    data.append(y)
    design.append(X)

import time 
GD = PYG.GLMdenoise(design, data, stim_dur=0.5, tr=0.764, n_jobs=1)
start = time.time()
GD.run()
print(f'Fit took {time.time()-start} seconds!')

# plot pseudo T statistics
brain = np.zeros(len(GD.mean_mask))
brain[GD.mean_mask] = GD.pseudo_t_stats.mean(0)
brain = brain.reshape(*dims[:-1], order='F')

from glmdenoise.utils.make_image_stack import make_image_stack
stack = make_image_stack(brain)

plt.imshow(stack)
