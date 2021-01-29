import time
import numpy as np

default_params = {
    'numforhrf': 50,
    'hrfthresh': 0.5,
    'hrffitmask': 1,
    'R2thresh': 0,
    'hrfmodel': 'optimise',
    'n_jobs': 1,
    'n_pcs': 20,
    'n_boots': 100,
    'extra_regressors': False,
    'wantlibrary': 1,
    'wantglmdenoise': 1,
    'wantfracridge': 0,
    'chunklen': 45000,
    'wantfileoutputs': [1, 1, 1, 1],
    'wantmemoryoutputs': [0, 0, 0, 1],
    'wantpercentbold': 1,
    'wantlss': 0,
    'brainthresh': [99, 0.1],
    'brainR2': [],
    'brainexclude': False,
    'pcR2cutoff': [],
    'pcR2cutoffmask': 1,
    'pcstop': 1.05,
    'fracs': np.linspace(1, 0.05, 20),
    'wantautoscale': 1,
    'seed': time.time(),
    'suppressoutput': 0,
    'lambda': 0
}
