from glmdenoise.makeimagestack import makeimagestack
import seaborn
from matplotlib import pyplot as plt
from ww import f
from os.path import join
import numpy


class Report(object):
    """Html report and figures
    """


    def __init__(self):
        self.blocks = []
        self.toc = []

    def add_image(self, name):
        """Add html for a figure
        
        Args:
            name (str): Name of figure
        """

        fpath = self.filepath_for(name)
        block_id = self.id_for(name)
        html = f('<h3 id="{block_id}">{name}</h3><img src="{fpath}" />\n')
        self.blocks.append(html)
        self.toc.append(name)

    def add_toc(self):
        """Add table of contents (list with links) to top of report
        """

        html = '<h2>Table of Contents</h2>\n'
        html += '<ol>\n'
        for name in self.toc:
            block_id = self.id_for(name)
            html += f('<li><a href="#{block_id}">{name}</a></li>\n')
        html += '</ol>\n'
        self.blocks.insert(0, html)

    def save(self):
        """Add title etc then save html to file
        """

        self.add_toc()
        self.blocks.insert(0, '<h1>GLMdenoise</h1>\n')
        with open('report.html', 'w') as html_file:
            for block in self.blocks:
                html_file.write(block + '\n')

    def plot_hrf(self, hrf1, hrf2, tr=2.0, title='Hemodynamic Reponse Function'):
        """Line plot of initial and estimated HRF
        
        Args:
            hrf1 (ndarray): hrf vector (1D)
            hrf2 (ndarray): another hrf vector (1D)
            tr (float, optional): Repetition time. Defaults to 2.0
            title (str, optional): Defaults to 'Hemodynamic Reponse Function'.
        """

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        t = numpy.arange(0, hrf1.size * tr, tr)
        ax.plot(t, hrf1, label='Initial HRF')
        ax.plot(t, hrf2, label='Estimated HRF')
        ax.axhline(0)
        ax.legend(loc='upper right')
        ax.set(xlabel='Time from condition onset (s)', ylabel='Response')
        fig.savefig(self.filepath_for(title))
        self.add_image(title)

    def plot_noise_regressors_cutoff(self, r2, n_noise_regressors, title):
        """Line plot of model fit by number of noise regressors used
        
        Args:
            r2 (ndarray): r-squared value for model fit for each number of 
                regressors included(1D)
            n_noise_regressors (int): The number of regressors chosen
            title (str): Name of plot
        """

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        max_nregressors = r2.shape[0]
        ax.plot(r2)
        ax.scatter(n_noise_regressors, r2[n_noise_regressors]) 
        ax.set_xticks(range(max_nregressors))
        ax.set_title(title)
        ax.set(xlabel='# noise regressors', ylabel='Median R2')
        fig.savefig(self.filepath_for(title))
        self.add_image(title)

    def plot_image(self, imgvector, title='no title', dtype='mask'):
        """Plot slices of a 3D image in a grid.

        Uses the spatial dimensions set on the report instance. 
        (Report.spatialdims)
        
        Args:
            imgvector (ndarray): Voxel values in a vector (1D)
            title (str, optional): Name of the plot. Defaults to 'no title'.
            dtype (str, optional): How to scale the data. Defaults to 'mask'.
        """

        # dtype= mask, range, scaled, percentile, custom
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        stack = makeimagestack(imgvector.reshape(self.spatialdims))
        ax.imshow(stack)
        fig.savefig(self.filepath_for(title))
        self.add_image(title)

    def filepath_for(self, name):
        return join('figures', self.id_for(name) + '.png')

    def id_for(self, name):
        return name.replace(' ', '_')

    def plot_scatter_sparse(self, data, xlabel, ylabel, title, crosshairs=False):
        """Scatter plot using max 1000 points for each series
        
        Args:
            data (list): List of (x, y) tuple data series   
            xlabel (str): Label on x-axis
            ylabel (str): Label on y-axis
            title (str): Name of plot
            crosshairs (bool, optional): Display horizontal and vertical
                lines through 0. Defaults to False.
        """

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        for x, y in data:
            nsample = min(1000, x.size)
            subset = numpy.random.choice(x.size, nsample, replace=False)
            ax.scatter(x[subset], y[subset])
        ax.set(xlabel=xlabel, ylabel=ylabel)
        fig.savefig(self.filepath_for(title))
        self.add_image(title)