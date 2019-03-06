from pprint import pprint


def run_files(bold_files, event_files, tr):
    """Run glmdenoise on the provided image and event files
    
    Args:
        bold_files (list): List of filepaths to .nii bold files
        event_files (list): List of filepaths to .tsv event files
        tr (float): Repetition time used across scans
    """

    print('## run_files ##')
    print('TR='+str(tr))
    pprint(bold_files)
    pprint(event_files)
