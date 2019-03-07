from glmdenoise.report import Report
from glmdenoise.select_voxels_nr_selection import select_voxels_nr_selection
from glmdenoise.select_noise_regressors import select_noise_regressors
import numpy

## fake data
nx, ny, nz, max_nregressors = 11, 12, 13, 20
nvoxels = nx * ny * nz
brain = numpy.random.rand(nx, ny, nz) - 0.5
pcR2 = numpy.repeat(brain[:,:,:, numpy.newaxis], max_nregressors, axis=3)
nr_range = numpy.arange(1, max_nregressors+1)
pcR2 = pcR2 * (nr_range/(nr_range + 0.25))

report = Report()
report.spatialdims = (nx, ny, nz)

## remove spatial dimensions
r2_voxels_nrs = pcR2.reshape([nvoxels, max_nregressors])

## voxels to use to evaluate solutions
voxels_nr_selection = select_voxels_nr_selection(r2_voxels_nrs)

## get the median model fit across voxels we just chose
r2_nrs = numpy.median(r2_voxels_nrs[voxels_nr_selection, :], 0)

## evaluate the solutions
n_noise_regressors = select_noise_regressors(r2_nrs)
print(n_noise_regressors)

title = 'Noise regressor selection'
report.plot_noise_regressors_cutoff(r2_nrs, n_noise_regressors,
    title='chosen number of regressors')

report.plot_image(brain[:], title)    

report.save()

#report.plot_hrf(hrfknobs, modelmd, title)
# for p in range(1, max_nregressors):
#     report.plot_scatter_sparse(
#         [
#             (r2_voxels_nrs[:, 0], r2_voxels_nrsr[:, p]),
#             (r2_voxels_nrs[pcvoxels, 0], r2_voxels_nrsr[pcvoxels, p]),
#         ],
#         xlabel='Cross-validated R^2 (0 PCs)',
#         ylabel='Cross-validated R^2 ({p} PCs)',
#         title='PCscatter{p}',
#         crosshairs=True,
#     )

# ## plot voxels for noise regressor selection
# title = 'Noise regressor selection'
# report.plot_noise_regressors_cutoff(r2_nrs, n_noise_regressors,
#     title='chosen number of regressors')
# report.plot_image(voxels_nr_selection, title)

# 

