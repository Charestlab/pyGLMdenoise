"""
GLMdenoise in python

"""
from glmdenoise.utils.normalisemax import normalisemax
from glmdenoise.utils.getcanonicalhrf import getcanonicalhrf
from glmdenoise.select_voxels_nr_selection import select_voxels_nr_selection
from glmdenoise.select_noise_regressors import select_noise_regressors
from glmdenoise.report import Report
import numpy


def run_data(design, data, tr, stimdur=0.5):
    # hrfmodel='optimise',hrfknobs=None,opt=None,figuredir=None



    ## fake output from step 6 
    nx, ny, nz, max_nregressors = 3, 4, 5, 20
    spatialdims = (nx, ny, nz)
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

    ##########################################################################
    ## Figures
    ##########################################################################

    ## start a new report with figures
    report = Report()
    report.spatialdims = (nx, ny, nz)

    ## plot solutions
    title = 'HRF fit'
    report.plot_hrf(hrfknobs, modelmd, tr, title)
    report.plot_image(hrffitvoxels, title)    

    for p in range(1, max_nregressors):
        report.plot_scatter_sparse(
            [
                (r2_voxels_nrs[:, 0], r2_voxels_nrsr[:, p]),
                (r2_voxels_nrs[pcvoxels, 0], r2_voxels_nrsr[pcvoxels, p]),
            ],
            xlabel='Cross-validated R^2 (0 PCs)',
            ylabel='Cross-validated R^2 ({p} PCs)',
            title='PCscatter{p}',
            crosshairs=True,
        )

    ## plot voxels for noise regressor selection
    title = 'Noise regressor selection'
    report.plot_noise_regressors_cutoff(r2_nrs, n_noise_regressors,
        title='chosen number of regressors')
    report.plot_image(voxels_nr_selection, title)

    ## various images
    report.plot_image(results.meanvol, 'Mean volume')
    report.plot_image(results.noisepool, 'Noise Pool')
    report.plot_image(opt.brainexclude, 'Noise Exclude')
    report.plot_image(opt.hrffitmask, 'HRFfitmask')
    report.plot_image(opt.pcR2cutoffmask, 'PCmask')

    for n in range(size(results.pcR2, dimdata+1)):
        report.plot_image(results.pcR2[:, n], 'PCcrossvalidation%02d', dtype='range')
        report.plot_image(results.pcR2[:, n], 'PCcrossvalidationscaled%02d', dtype='scaled')

    report.plot_image(results.R2, 'FinalModel')
    for r in range(nruns):
        report.plot_image(results.R2run[r], 'FinalModel_run%02d')

    report.plot_image(results.signal,'SNRsignal.png', dtype='percentile')
    report.plot_image(results.noise, 'SNRnoise', dtype='percentile')
    report.plot_image(results.SNR, 'SNR', dtype='percentile')
  
    ## Signal to noise
    report.plot_scatter_sparse(
        [
            (results.SNRbefore[:], results.SNRafter[:]),
            (results.SNRbefore(pcvoxels), results.SNRafter(pcvoxels)),
        ],
        xlabel='SNR (before denoising)',
        ylabel='SNR (after denoising)',
        title='SNRcomparebeforeandafter',
    )
    datagain = ((results.SNRafter / results.SNRbefore) ** 2 - 1) * 100
    report.plot_scatter_sparse(
        [
            (results.SNRbefore[:], datagain[:]),
            (results.SNRbefore[pcvoxels], datagain[pcvoxels]),
        ],
        xlabel='SNR (before denoising)',
        ylabel='Equivalent amount of data gained (%)',
        title='SNRamountofdatagained',
    )

    ## PC weights
    thresh = prctile(numpy.abs(results.pcweights[:]), 99)
    for p in range(1, size(results.pcweights, dimdata+1)):
        for q in range(1, size(results.pcweights, dimdata+2)):
            report.plot_image(
                results.pcweights[p, q],
                'PCmap_run%02d_num%02d.png',
                dtype='custom',
                drange=[-thresh, thresh]
            )

    # stores html report
    report.save()
