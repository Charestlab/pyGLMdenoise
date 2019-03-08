"""
Estimate HRF for each voxel
"""
import numpy as np
import pandas as pd
from nistats.design_matrix import make_design_matrix


stimdur = 0.5
TR = 0.764
n_scans = data.shape[1]
frame_times = np.arange(n_scans) * TR

data = np.random.random((100, 10))

# Load onsets and item presented
onsets = pd.read_csv('data/sub-1_WM_onset.tsv', sep='\t')['onset'].values
items = pd.read_csv('data/sub-1_WM_onset.tsv', sep='\t')['item'].values
n_events = len(onsets)

# Create design matrix
events = pd.DataFrame()
events['duration'] = [stimdur]*n_events
events['onset'] = onsets
events['trial_type'] = np.arange(n_events)+1 # single trial

X = make_design_matrix(
    frame_times, events, hrf_model='glover', drift_order=2, drift_model='polynomial')


def dark_magic(tada):
    """
    """
