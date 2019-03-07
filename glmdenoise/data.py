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
    report.plot_hrf(hrfknobs, modelmd, title)
    report.plot_image(hrffitvoxels, title)    

    ## plot voxels for noise regressor selection
    title = 'Noise regressor selection'
    report.plot_noise_regessors_cutoff(r2_nrs, title)
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


#     % make figure illustrating selection of number of PCs
#     figureprep([100 100 400 400]); hold on;
#     plot(0:opt.numpcstotry,xvaltrend,'r.-');
#     set(scatter(results.pcnum,xvaltrend(1+results.pcnum),100,'ro'),'LineWidth',2);
#     set(gca,'XTick',0:opt.numpcstotry);
#     xlabel('Number of PCs');
#     ylabel('Cross-validated R^2 (median across voxels)');
#     title(sprintf('Selected PC number = %d',results.pcnum));
#     figurewrite('PCselection',[],[],figuredir);
  
#     % make figure showing scatter plots of cross-validated R^2
#     rng = [min(results.pcR2(:)) max(results.pcR2(:))];
#     for p=1:opt.numpcstotry
#       temp = squish(results.pcR2,dimdata);  % voxels x 1+pcs
#       figureprep([100 100 500 500]); hold on;
#       scattersparse(temp(:,1),temp(:,1+p),20000,0,36,'g','.');
#       scattersparse(temp(pcvoxels,1),temp(pcvoxels,1+p),20000,0,36,'r','.');
#       axis([rng rng]); axissquarify; axis([rng rng]); 
#       straightline(0,'h','y-');
#       straightline(0,'v','y-');
#       xlabel('Cross-validated R^2 (0 PCs)');
#       ylabel(sprintf('Cross-validated R^2 (%d PCs)',p));
#       title(sprintf('Number of PCs = %d',p));
#       figurewrite(sprintf('PCscatter%02d',p),[],[],figuredir);
#     end
  
#   % write out SNR comparison figures (first figure)
#   if opt.numboots ~= 0
#     rng = [min([results.SNRbefore(:); results.SNRafter(:)]) max([results.SNRbefore(:); results.SNRafter(:)])];
#     if ~all(isfinite(rng))  % hack to deal with cases of no noise estimate
#       rng = [0 1];
#     end
#     figureprep([100 100 500 500]); hold on;
#     scattersparse(results.SNRbefore(:),results.SNRafter(:),20000,0,36,'g','.');
#     if ~wantbypass
#       scattersparse(results.SNRbefore(pcvoxels),results.SNRafter(pcvoxels),20000,0,36,'r','.');
#     end
#     axis([rng rng]); axissquarify; axis([rng rng]);
#     xlabel('SNR (before denoising)');
#     ylabel('SNR (after denoising)');
#     figurewrite(sprintf('SNRcomparebeforeandafter'),[],[],figuredir);
#   end
  
#   % write out SNR comparison figures (second figure)
#   if opt.numboots ~= 0
#     datagain = ((results.SNRafter./results.SNRbefore).^2 - 1) * 100;
#     figureprep([100 100 500 500]); hold on;
#     scattersparse(results.SNRbefore(:),datagain(:),20000,0,36,'g','.');
#     if ~wantbypass
#       scattersparse(results.SNRbefore(pcvoxels),datagain(pcvoxels),20000,0,36,'r','.');
#     end
#     ax = axis; axis([rng ax(3:4)]);
#     xlabel('SNR (before denoising)');
#     ylabel('Equivalent amount of data gained (%)');
#     figurewrite(sprintf('SNRamountofdatagained'),[],[],figuredir);
#   end
  
#   % write out maps of pc weights
#   thresh = prctile(abs(results.pcweights(:)),99);
#   for p=1:size(results.pcweights,dimdata+1)
#     for q=1:size(results.pcweights,dimdata+2)
#       temp = subscript(results.pcweights,[repmat({':'},[1 dimdata]) {p} {q}]);
#       imwrite(uint8(255*makeimagestack(opt.drawfunction(temp),[-thresh thresh])),cmapsign(256), ...
#               fullfile(figuredir,'PCmap',sprintf('PCmap_run%02d_num%02d.png',q,p)));
#     end
#   end

# end
