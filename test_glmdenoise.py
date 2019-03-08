from os.path import join
import nibabel as nib
import numpy as np
import scipy.io as sio
from statsmodels.regression.linear_model import OLS
from glmdenoise.utils.stimMat import constructStimulusMatrices
from glmdenoise.utils.getcanonicalhrf import getcanonicalhrf
from glmdenoise.utils.get_poly_matrix import get_poly_matrix
from glmdenoise.utils.optimiseHRF import convolveDesign
import matplotlib.pyplot as plt
import seaborn as sns

basedir = '/media/charesti-start/data/irsa-fmri'
bidsdir = join(basedir, 'BIDS')
mridatadir = join(bidsdir, 'derivatives')
fmriprepdir = join(mridatadir, 'fmriprep')

task = 'irsa'
sub = 1
ses = 1
nruns = 9
duration = 1
tr = 2
ntime = 216

design_template = join(
    bidsdir,
    f'sub-{sub}', f'ses-{ses}', 'func', 'sub-{}_ses-{}_task-{}_design.mat')
epi_template = 'sub-{}_ses-{}_task-{}_run-0{}_bold_space-MNI152NLin2009cAsym_preproc.nii.gz'
# load the design
tmpdesign = sio.loadmat(design_template.format(sub, ses, task))['design'][0]

# make it a list of arrays instead of this tuple thing
design = []
for run in range(nruns):
    design.append(tmpdesign[run])

data = []
for run in range(nruns):
    print('loading data for run {}\n'.format(run+1))
    thisEPI = join(fmriprepdir, f'sub-{sub}', f'ses-{ses}',
                   'func', epi_template.format(sub, ses, task, run+1))
    nim = nib.load(thisEPI)
    img = nim.get_data()
    x, y, z, ntime = img.shape
    img = img.reshape((x*y*z, ntime))
    data.append(np.single(img))
numruns = len(design)

rundes = design[0]
hrfknobs = getcanonicalhrf(duration, tr)
numinhrf = len(hrfknobs)
ntime, numcond = rundes.shape
postnumlag = numinhrf-1


stimmat = constructStimulusMatrices(
    rundes.T, prenumlag=0, postnumlag=postnumlag)

convdesign = []
convdesignpre = []
for p in range(numruns):
    # expand design matrix using delta functions
    ntime = design[p].shape[0]

    stimmat = constructStimulusMatrices(
        design[p].T, prenumlag=0, postnumlag=postnumlag)  # time x L*conditions

    # time*L x conditions
    convdesignpre.append(stimmat.reshape(numinhrf*ntime, numcond))

    convdes = convolveDesign(design[p], hrfknobs)

    convdes = get_poly_matrix(convdes.shape[0], [0, 1, 2, 3]) * convdes
    convdesign.append(convdes)


sns.heatmap(stimmat)
