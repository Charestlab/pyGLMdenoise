"""
GLMdenoise in python

"""
from glmdenoise.utils.normalisemax import normalisemax
from glmdenoise.utils.getcanonicalhrf import getcanonicalhrf
from glmdenoise.select_voxels_nr_selection import select_voxels_nr_selection
from glmdenoise.select_noise_regressors import select_noise_regressors
import numpy


def GLMdenoisedata(design,data,stimdur,tr):
    # hrfmodel='optimise',hrfknobs=None,opt=None,figuredir=None

    # fake output from step 6 
    nx, ny, nz, max_nregressors = 3, 4, 5, 20
    nvoxels = nx * ny * nz
    pcR2 = numpy.zeros([nx, ny, nz, max_nregressors])

    ## Step 7: Select number of noise regressors
    r2_voxels_nrs = pcR2.reshape([nvoxels, max_nregressors])
    voxels_nr_selection = select_voxels_nr_selection(r2_voxels_nrs)
    r2_nrs = numpy.median(r2_voxels_nrs[voxels_nr_selection, :], 0)
    n_noise_regressors = select_noise_regressors(r2_nrs)

    ######################### DEAL WITH INPUTS, ETC.
    
    # default hrfmodel is optimise
    
    # if hrfknobs is None:
    #     if hrfmodel == 'fir':
    #         hrfknobs = 20
    #     else:
    #         hrfknobs = normalisemax(getcanonicalhrf(stimdur,tr).T)
    
    # if opt is None:
    #     opt = dict()
    
    # if figuredir is None:
    #     figuredir = 'GLMdenoisefigures'
    
    # # massage input
    # if not isinstance(design, dict):
    #     design = {design}
    
    # # make sure the data is in the right format
    # if not isinstance(data,dict):
    #     data[0] = data
    
    # # reduce precision to single
    # for p, dataset in enumerate(data):
    #     if not isinstance(dataset, np.float32):
    #         print('***\n\n\n##########\n GLMdenoisedata: converting data in run {0} to single format (consider doing this before the function call to reduce memory usage). \n\n\n##########\n***\n'.format(p))
    #         data[p] = np.single(data[p])   
    
    # # do some error checking
    # if any(np.isfinite(data[1])):
    #     print('***\n\n\n##########\n GLMdenoisedata: WARNING: we checked the first run and found some non-finite values (e.g. NaN, Inf). unexpected results may occur due to non-finite values. please fix and re-run GLMdenoisedata. \n\n\n##########\n***\n')
    
    # if hrfknobs is None:
    #     if hrfmodel == 'fir':
    #         hrfknobs = 20
    #     else:
    #         hrfknobs = normalisemax(getcanonicalhrf(stimdur,tr).T)
    
    # if opt is None:
    #     opt = dict()
    
    # if figuredir is None:
    #     figuredir = 'GLMdenoisefigures'
    
    # # massage input
    # if not isinstance(design, dict):
    #     design = {design}
    
    # # make sure the data is in the right format
    # if not isinstance(data,dict):
    #     data[0] = data
    
    # # reduce precision to single
    # for p, dataset in enumerate(data):
    #     if not isinstance(dataset, np.float32):
    #         print('***\n\n\n##########\n GLMdenoisedata: converting data in run {0} to single format (consider doing this before the function call to reduce memory usage). \n\n\n##########\n***\n'.format(p))
    #         data[p] = np.single(data[p])   
    
    # # do some error checking
    # if any(np.isfinite(data[1])):
    #     print('***\n\n\n##########\n GLMdenoisedata: WARNING: we checked the first run and found some non-finite values (e.g. NaN, Inf). unexpected results may occur due to non-finite values. please fix and re-run GLMdenoisedata. \n\n\n##########\n***\n')
    