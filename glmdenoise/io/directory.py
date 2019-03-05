from glmdenoise.io.files import run_files
from glmdenoise.io.bids import BidsDirectory


def run_bids_directory(directory='.', sub_num=None, sub=None, task=None):
    bids = BidsDirectory(directory)
    return run_bids(bids, sub_num=None, sub=None, task=None)


def run_bids(bids, sub_num=None, sub=None, task=None):
    pass