from glmdenoise.report import Report
from os import mkdir
import nibabel
import numpy
import os


class Output(object):
    """Service which stores results in files
    """

    def __init__(self):
        pass

    def determine_location(self, sample_file):
        """Supply one input file to initialize output
        
        Args:
            sample_file (str): Path to nifti file for one run
        """

        datadir = os.path.dirname(sample_file)
        self.outdir = os.path.join(datadir, 'glmdenoise')

    def determine_location_in_bids(self, bids, sub, ses, task):
        pass

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
        fpath = os.path.join(self.outdir, name + '.npy')
        numpy.save(fpath, var)

    def ensure_directory(self):
        """Make sure that the output directory exist.

        Tries to create the output directory if it isn't there yet.
        """
        if not os.path.isdir(self.outdir):
            mkdir(self.outdir)
