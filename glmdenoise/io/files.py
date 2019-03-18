from glmdenoise.data import run_data
from pprint import pprint
import nibabel
import pandas


def run_files(bold_files, event_files, tr):
    """Run glmdenoise on the provided image and event files
    
    Args:
        bold_files (list): List of filepaths to .nii bold files
        event_files (list): List of filepaths to .tsv event files
        tr (float): Repetition time used across scans
    """

    msg = 'need same number of image and event files'
    assert len(bold_files) == len(event_files), msg
    data = [nibabel.load(f).get_data() for f in bold_files]
    design = [pandas.read_csv(f, delimiter='\t') for f in event_files]
    run_data(data, design, tr)
