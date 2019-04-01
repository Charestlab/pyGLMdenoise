from glmdenoise.report import Report
from os import mkdir
import nibabel
import numpy
import os


class Output(object):
    """Service which stores results in files
    """

    def __init__(self):
        self.bids = None

    def determine_location(self, sample_file):
        """Supply one input file to initialize output
        
        Args:
            sample_file (str): Path to nifti file for one run
        """

        datadir = os.path.dirname(sample_file)
        self.outdir = os.path.join(datadir, 'glmdenoise')

    def determine_location_in_bids(self, bids, sub, ses, task):
        self.bids = bids
        self.entities = {'sub': sub, 'ses': ses, 'task': task}
        subdir = 'sub-{}'.format(sub)
        sesdir = 'ses-{}'.format(ses)
        subdir = os.path.join('derivatives', 'glmdenoise', subdir, sesdir)
        self.outdir = os.path.join(bids.root, subdir)

    def file_path(self, tag, ext):
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
        return Report()

    def save_image(self, imageArray, name):
        pass

    def save_variable(self, var, name):
        """Store non-brain-image data in a file
        
        Args:
            var: The value to store
            name (str): The name of the variable
        """
        self.ensure_directory()
        numpy.save(self.file_path(name, 'npy'), var)

    def ensure_directory(self):
        """Make sure that the output directory exist.

        Tries to create the output directory if it isn't there yet.
        """
        if not os.path.isdir(self.outdir):
            mkdir(self.outdir)
