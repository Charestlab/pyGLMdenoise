"""
GLMdenoise in python

"""
from glmdenoise.utils.normalisemax import normalisemax
from glmdenoise.utils.getcanonicalhrf import getcanonicalhrf
from glmdenoise.select_voxels_nr_selection import select_voxels_nr_selection
from glmdenoise.select_noise_regressors import select_noise_regressors
from glmdenoise.makeimagestack import makeimagestack
import numpy, seaborn


def run_data(design, data, tr, stimdur=0.5):
    # hrfmodel='optimise',hrfknobs=None,opt=None,figuredir=None

    ## fake output from step 6 
    nx, ny, nz, max_nregressors = 3, 4, 5, 20
    nvoxels = nx * ny * nz
    brain = numpy.random.rand(nx, ny, nz) - 0.5
    pcR2 = numpy.repeat(brain[:,:,:, numpy.newaxis], max_nregressors, axis=3)
    nr_range = numpy.arange(1, max_nregressors+1)
    pcR2 = pcR2 * (nr_range/(nr_range + 0.25))

    ##########################################################################
    ## Step 7: Select number of noise regressors
    ##########################################################################

    ## remove spatial dimensions
    r2_voxels_nrs = pcR2.reshape([nvoxels, max_nregressors])

    ## voxels to use to evaluate solutions
    voxels_nr_selection = select_voxels_nr_selection(r2_voxels_nrs)

    ## get the median model fit across voxels we just chose
    r2_nrs = numpy.median(r2_voxels_nrs[voxels_nr_selection, :], 0)

    ## evaluate the solutions
    n_noise_regressors = select_noise_regressors(r2_nrs)
    print(n_noise_regressors)

    ## plot solutions
    ax = seaborn.lineplot(data=r2_nrs)
    ax.scatter(n_noise_regressors, r2_nrs[n_noise_regressors]) 
    ax.set_xticks(range(max_nregressors))
    ax.set_title('chosen number of regressors')
    ax.set(xlabel='# noise regressors', ylabel='Median R2')

    ## plot voxels for noise regressor selection
    import matplotlib.pyplot as plt 
    stack = makeimagestack(voxels_nr_selection.reshape(nx, ny, nz))
    ax = plt.imshow(stack)
    

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
    