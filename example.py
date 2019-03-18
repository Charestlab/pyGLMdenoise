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
from glmdenoise.utils.optimize_hrf import optimiseHRF
from glmdenoise.utils.make_poly_matrix import *
warnings.simplefilter(action="ignore", category=FutureWarning)

def format_time(time):
    return f'{int(time//60)} minutes and {time-(time//60)*60:.2f} seconds'

"""
Load data
"""

fmri_folder = '/home/dlindh/AMS_AB_RSA/preproc/fmriprep/sub-01'

runs = glob.glob(os.path.join(fmri_folder, 'ses*', 'func', '*preproc*nii.gz'))
runs.sort()
behavior = glob.glob(os.path.join(fmri_folder, 'ses*', 'func', '*_events.tsv'))
behavior.sort()

runs = compress(runs, np.arange(len(runs)) != 1)
behavior = compress(behavior, np.arange(len(behavior)) != 1)

data = []
design = []
hrf = np.load('hrf.npy')    
for i, (run, event) in enumerate(zip(runs, behavior)):
    print(f'run {i}')
    y = nib.load(run).get_data()
    data.append(y)

    stimdur = 0.5
    TR = 0.764
    n_volumes = y.shape[-1]

    # Load onsets and item presented
    onsets = pd.read_csv(event, sep='\t')["onset"].values
    items = pd.read_csv(event, sep='\t')["item"].values
    n_events = len(onsets)

    # Create design matrix
    events = pd.DataFrame()
    events["duration"] = [stimdur] * n_events
    events["onset"] = onsets
    events["trial_type"] = items

    X = make_design(events, TR, n_volumes, hrf=hrf)
    design.append(X)

# where you can optimize the HRF if needed.
# polynomials = [make_poly_matrix(y.shape[-1], np.arange(5)) for y in data]
# polynomials = [make_project_matrix(p) for p in polynomials]
# convolved_design, optimized_hrf = optimiseHRF(design, data, TR, hrf, polynomials)


"""
Run GLMdenoise
"""

import time 
gd = PYG.GLMdenoise(design, data, tr=0.764, n_jobs=1)
start = time.time()
gd.fit()
print(f'Fit took {format_time(time.time()-start)}!')

gd.plot_figures()