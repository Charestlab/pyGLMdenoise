import glmdenoise
from os import makedirs
import nibabel
import numpy
import os


class Output(object):
    """Service which stores results in files
    """

    def __init__(self):
        self.bids = None
        self.outdir = 'glmdenoise'

    def configure_from(self, sample_file):
        """Supply one input file to initialize output
        
        Args:
            sample_file (str): Path to nifti file for one run
        """

        datadir = os.path.dirname(sample_file)
        self.img = nibabel.load(sample_file)
        if self.outdir == 'glmdenoise':
            # still default, update
            self.outdir = os.path.join(datadir, 'glmdenoise')

    def fit_bids_context(self, bids, sub, ses, task):
        """Provide BIDS context including subject, session and task
        
        Args:
            bids (glmdenoise.io.bids.BidsDirectory): Bids directory object
            sub (str): subject
            ses (str): session
            task (str): task name
        """

        self.bids = bids
        self.entities = {'sub': sub, 'ses': ses, 'task': task}
        subdir = 'sub-{}'.format(sub)
        sesdir = 'ses-{}'.format(ses)
        subdir = os.path.join('derivatives', 'glmdenoise', subdir, sesdir)
        self.outdir = os.path.join(bids.root, subdir)

    def ensure_directory(self):
        """Make sure that the output directory exist.

        Tries to create the output directory if it isn't there yet.
        """
        if not os.path.isdir(self.outdir):
            makedirs(self.outdir)

    def safe_name(self, name):
        """Remove spaces etc so it can be used as html identifier
        
        Args:
            name (str): Title or name
        
        Returns:
            str: name without special characters
        """

        return name.replace(' ', '_')

    def file_path(self, tag, ext):
        """Obtain output file path for the given variable name and extension
        
        Args:
            tag (str): Variable to save
            ext (str): File extension without leading dot
        
        Returns:
            str: Absolute file path for file to save
        """

        if self.bids:
            if self.entities.get('ses'):
                templ = 'sub-{sub}_ses-{ses}_task-{task}_{tag}.{ext}'
            else:
                templ = 'sub-{sub}_task-{task}_{tag}.{ext}'
            fname = templ.format(**dict(tag=tag, ext=ext, **self.entities))
            return os.path.join(self.outdir, fname)
        else:
            return os.path.join(self.outdir, tag + '.' + ext)

    def create_report(self):
        """Return a Report object configured to use this Output object
        
        Returns:
            glmdenoise.report: Report object with methods to create figures
        """

        report = glmdenoise.report.Report()
        report.use_output(self)
        return report

    def save_image(self, imageArray, name):
        """Store brain image data in a nifti file using nibabel
        
        Args:
            imageArray (ndarray): data
            name (str): The name of the variable
        """
        img = nibabel.Nifti1Image(
            imageArray,
            self.img.get_affine(),
            header=self.img.header
        )
        img.to_filename(self.file_path(name, 'nii'))

    def save_variable(self, var, name):
        """Store non-brain-image data in a file
        
        Args:
            var: The value to store
            name (str): The name of the variable
        """
        self.ensure_directory()
        numpy.save(self.file_path(name, 'npy'), var)

    def save_text(self, text, name, ext):
        """Store text file
        
        Args:
            text (str): string content
            name (str): filename
            ext (str): file extension
        """

        with open(self.file_path(name, ext), 'w') as text_file:
            text_file.write(text)

    def save_figure(self, figure, name):
        """Save the figure object as a png file
        
        Args:
            figure (matplotlib.figure.Figure): Figure object
            name (str): name of the plot
        
        Returns:
            str: absolute filepath to new png file
        """

        fpath = self.file_path(name, 'png')
        figure.save_figure(fpath)
        return fpath