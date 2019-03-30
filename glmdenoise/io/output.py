from glmdenoise.report import Report
from os import mkdir
import nibabel
import numpy
import os


class Output(object):

    def __init__(self):
        pass

    def determine_location(self, sample_file):
        datadir = os.path.dirname(sample_file)
        self.outdir = os.path.join(datadir, 'glmdenoise')

    def determine_location_in_bids(self, bids, sub, ses, task):
        pass

    def create_report(self):
        return Report()

    def save_image(self, imageArray, name):
        pass

    def save_variable(self, var, name):
        self.ensure_directory()
        fpath = os.path.join(self.outdir, name + '.npy')
        numpy.save(fpath, var)

    def ensure_directory(self):
        if not os.path.isdir(self.outdir):
            mkdir(self.outdir)
