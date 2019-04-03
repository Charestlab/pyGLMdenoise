from glmdenoise.pyGlmdenoise import GLMdenoise
from glmdenoise.io.output import Output
from glmdenoise.io.input import load_nifti
from pprint import pprint
import nibabel
import pandas


def run_files(bold_files, event_files, tr, out=None):
    """Run glmdenoise on the provided image and event files

    Args:
        bold_files (list): List of filepaths to .nii bold files
        event_files (list): List of filepaths to .tsv event files
        tr (float): Repetition time used across scans
    """

    msg = 'need same number of image and event files'
    assert len(bold_files) == len(event_files), msg
    if out is None:
        out = Output()
    out.configure_from(sample_file=bold_files[0])
    data = [load_nifti(f) for f in bold_files]
    design = [pandas.read_csv(f, delimiter='\t') for f in event_files]
    gd = GLMdenoise(params={'hrfmodel': 'assume'})
    gd.fit(design, data, tr)
    gd.plot_figures(out.create_report())
    for image_name in ['pseudo_t_stats']:
        out.save_image(gd.results.get(image_name), image_name)
    for var_name in ['xval']:
        out.save_variable(gd.results.get(var_name), var_name)
