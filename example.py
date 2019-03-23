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

params = {}
params['hrf'] = normalisemax(getcanonicalhrf(stimdur, TR))
params['tr'] = TR
params['numforhrf'] = 50
params['hrfthresh'] = 0.5
params['hrffitmask'] = 1
params['R2thresh'] = 0
params['hrfmodel'] = 'optimise'  # 'assume'
params['extra_regressors'] = False

for i, (run, event) in enumerate(zip(runs, eventfs)):
    print('run {}'.format(i))
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

    # pass in the events data frame. the convolving of the HRF now
    # happens internally
    design.append(events)
    data.append(y)


gd = PYG.GLMdenoise(design, data, params, n_jobs=2)
start = time.time()
gd.fit()
fit_dur = format_time(time.time()-start)
print('Fit took {}!'.format(fit_dur)

gd.plot_figures()
