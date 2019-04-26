

def run_public(dataset, sub=None, task=None):
    """Download dataset by name and denoise it.

    Example: `glmdenoise ///workshops/nih-2017/ds000114`
    
    Args:
        dataset (str): Name of dataset, e.g. '///workshops/nih-2017/ds000114'
        subject (str, optional): BIDS identifier of one subject to run.
            Defaults to None, meaning all subjects.
        task (str, optional): Name of specific task to run.
            Defaults to None, meaning all tasks.
    """

    reqmsg = """
    You're trying to run GLMdenoise on a publicly available dataset.
    
    This requires:
    - datalad   >= 0.11.4       (pip install datalad)
    - git-annex >= 6.20180913   (on Ubuntu 19.04: apt install git-annex)
    """
    try:
        import datalad.api
    except ImportError:
        print(reqmsg)
        exit(65)
    datalad.api.install(source=dataset, path='~/datalad', recursive=True, get_data=True)
    # get_data=True
    # datalad install -r -s ///labs/gobbini/famface ~/dlfam
    # must generate path with dataset name
    print(dataset)