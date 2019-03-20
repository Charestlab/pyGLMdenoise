from glmdenoise.io.files import run_files
from glmdenoise.io.bids import BidsDirectory


def run_bids_directory(directory='.', sub_num=None, sub=None, task=None):
    """Run glmdenoise on a whole or part of a dataset in a BIDS directory

    Args:
        directory (str, optional): Root data directory containing BIDS.
            Defaults to '.'
        sub_num (int, optional): Number of one subject to run. 
            Defaults to None.
        sub (string, optional): BIDS identifier of one subject to run. 
            Defaults to None.
        task (string, optional): Name of specific task to run. 
            Defaults to None.
    """

    bids = BidsDirectory(directory)
    return run_bids(bids, sub_num=sub_num, sub=sub, task=task)


def run_bids(bids, sub_num=None, sub=None, task=None, ses=None):
    """Recursively run GLMdenoise on subjects, tasks, sessions

    This uses a bids object that is able to find data in the BIDS
    directory it represents.
    See `run_bids_directory` to call this with a directory path.
    
    Args:
        bids (glmdenoise.io.bids.BidsDirectory): Wrapper for pybids
        sub_num (int, optional): Number of one subject to run. 
            Defaults to None.
        sub (str, optional): BIDS identifier of one subject to run. 
            Defaults to None. 
        task (str, optional): Name of specific task to run. 
            Defaults to None. 
        ses (str, optional): Session identifier to run.
            Defaults to None, meaning run all sessions one by one 
    
    """
    if sub and task and ses:
        return run_bids_subset(bids, sub, task, ses)
    elif sub and task:
        sessions = bids.get_sessions_for_task_and_subject(task, sub)
        for ses in sessions:
            run_bids(bids, sub=sub, task=task, ses=ses)
        if sessions == []:
            run_bids_subset(bids, sub, task)
    elif sub:
        tasks = bids.get_tasks_for_subject(sub)
        for task in tasks:
            run_bids(bids, sub=sub, task=task)
    elif sub_num:
        sub = bids.subject_id_from_number(sub_num)
        assert sub, 'Could not match subject index to a subject ID'
        run_bids(bids, sub=sub)
    else:
        subs = bids.get_preprocessed_subjects_ids()
        for sub in subs:
            run_bids(bids, sub=sub)

def run_bids_subset(bids, sub, task, ses=None):
    """Run GLMdenoise on a subset of files in a BIDS dir defined by the args

    This uses a bids object that is able to find data in the BIDS
    directory it represents.
    See `run_bids_directory` to call this with a directory path.
    
    Args:
        bids (glmdenoise.io.bids.BidsDirectory): Wrapper for pybids
        sub (str): BIDS identifier of one subject to run. 
        task (str): Name of specific task to run. 
        ses (str, optional): Session identifier to run.
            Defaults to None, meaning there are no sessions. 
    
    """
    bold_files = bids.get_filepaths_bold_runs(sub, task, ses)
    if not bold_files:
        msg = 'No preprocessed runs found for subject {} task {} session {}'
        print(msg.format(sub, task, ses))
        return
    event_files = bids.get_filepaths_event_runs(sub, task, ses)
    metas = bids.get_metas_bold_runs(sub, task, ses)
    key = 'RepetitionTime'
    trs = [meta[key] for meta in metas if key in meta]
    assert trs, 'RepetitionTime not specified in metadata'
    assert len(set(trs)) == 1, 'RepetitionTime varies across runs'
    return run_files(bold_files, event_files, tr=trs[0])
