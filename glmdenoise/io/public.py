

def run_public(dataset, sub=None, task=None):
    """Download dataset by name and denoise it.
    
    Args:
        dataset (str): Name of dataset, e.g. 'ds000150'
        subject (str, optional): BIDS identifier of one subject to run.
            Defaults to None.
        task (str, optional): Name of specific task to run.
            Defaults to None.
    """
    print(dataset)