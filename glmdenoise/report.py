from glmdenoise.makeimagestack import makeimagestack
import seaborn
from matplotlib import pyplot as plt
from ww import f


class Report(object):

    def __init__(self):
        self.blocks = []
        self.toc = []

    def add_image(self, name):
        fname = 'afile.png'
        html = f('<img id="{name}" src="figures/{fname}" />')
        self.blocks.append(html)
        self.toc.append(name)

    def add_toc(self):
        html = ''
        html += '<ol>'
        for name in self.toc:
            html += f('<li><a href="#{name}">{name}</a></li>')
        html += '</ol>'
        self.blocks.insert(0, html)

    def save(self):
        self.add_toc()
        with open('report.html', 'w') as html_file:
            for block in self.blocks:
                html_file.write(block + '\n')

    def plot_hrf(self, hrf1, hrf2, title):
        pass

#   % make figure showing HRF
#   if ~isequal(hrfmodel,'fir') && length(hrfknobs) > 1
#     figureprep([100 100 450 250]); hold on;
#     numinhrf = length(hrfknobs);
#     h1 = plot(0:tr:(numinhrf-1)*tr,hrfknobs,'ro-');
#     h2 = plot(0:tr:(numinhrf-1)*tr,results.modelmd{1},'bo-');
#     ax = axis; axis([0 (numinhrf-1)*tr ax(3) 1.2]);
#     straightline(0,'h','k-');
#     legend([h1 h2],{'Initial HRF' 'Estimated HRF'});
#     xlabel('Time from condition onset (s)');
#     ylabel('Response');
#     figurewrite('HRF',[],[],figuredir);
#   end

    def plot_noise_regressors_cutoff(self, r2, n_noise_regressors, title):
        max_nregressors = r2.shape[1]
        ax = seaborn.lineplot(data=r2)
        ax.scatter(n_noise_regressors, r2[n_noise_regressors]) 
        ax.set_xticks(range(max_nregressors))
        ax.set_title(title)
        ax.set(xlabel='# noise regressors', ylabel='Median R2')

    def plot_image(self, imgvector, dtype='mask'):
        # dtype= mask, range, scaled, percentile, custom
        import matplotlib.pyplot as plt 
        stack = makeimagestack(imgvector.reshape(self.spatialdims))
        ax = plt.imshow(stack)

#   % write out image showing HRF fit voxels
#   if isequal(hrfmodel,'optimize') && ~isempty(results.hrffitvoxels)
#     imwrite(uint8(255*makeimagestack(opt.drawfunction(results.hrffitvoxels),[0 1])),gray(256),fullfile(figuredir,'HRFfitvoxels.png'));
#   end
#   % define a function that will write out R^2 values to an image file
#   imfun = @(results,filename) ...
#     imwrite(uint8(255*makeimagestack(opt.drawfunction(signedarraypower(results/100,0.5)),[0 1])),hot(256),filename);

#     % figure out bounds for the R^2 values
#     bounds = prctile(results.pcR2(:),[1 99]);
#     if bounds(1)==bounds(2)  % a hack to avoid errors in normalization
#       bounds(2) = bounds(1) + 1;
#     end

#     % define another R^2 image-writing function
#     imfunB = @(results,filename) ...
#       imwrite(uint8(255*makeimagestack(opt.drawfunction(signedarraypower(normalizerange(results,0,1,bounds(1),bounds(2)),0.5)),[0 1])),hot(256),filename);

    def plot_scatter_sparse(self, data, xlabel, ylabel, title, crosshairs=False):
        pass