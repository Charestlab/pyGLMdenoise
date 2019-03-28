from glmdenoise.pyGlmdenoise import GLMdenoise
from glmdenoise.io.output import Output
from pprint import pprint
import nibabel
import pandas


def run_files(bold_files, event_files, tr, bids=None):
    """Run glmdenoise on the provided image and event files

    Args:
        bold_files (list): List of filepaths to .nii bold files
        event_files (list): List of filepaths to .tsv event files
        tr (float): Repetition time used across scans
    """

    msg = 'need same number of image and event files'
    assert len(bold_files) == len(event_files), msg
    output = Output(bold_files[0], bids)
    data = [nibabel.load(f).get_data() for f in bold_files]
    design = [pandas.read_csv(f, delimiter='\t') for f in event_files]
    gd = GLMdenoise(design, data, tr)
    gd.fit()
    gd.plot_figures(output.create_report())
