

def run_public(dataset, sub=None, task=None):
    """Download dataset by name and denoise it.
    
    Args:
        dataset (str): Name of dataset, e.g. '///openfmri/ds000006'
        subject (str, optional): BIDS identifier of one subject to run.
            Defaults to None, meaning all subjects.
        task (str, optional): Name of specific task to run.
            Defaults to None, meaning all tasks.
    """

    import datalad.api
    datalad.api.install(source=dataset, path='~/datalad', recursive=True)
    # get_data=True
    # datalad install -r -s ///labs/gobbini/famface ~/dlfam
    # must generate path with dataset name
    print(dataset)