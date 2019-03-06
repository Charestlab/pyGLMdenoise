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


def run_bids(bids, sub_num=None, sub=None, task=None):
    if sub and task:
        bold_files = bids.get_filepaths_bold_runs(sub, task)
        if not bold_files:
            msg = 'No preprocessed runs found for subject {} task {}'
            print(msg.format(sub, task))
            return
        event_files = bids.get_filepaths_event_runs(sub, task)
        metas = bids.get_metas_bold_runs(sub, task)
        trs = [meta['RepetitionTime'] for meta in metas]
        assert trs, 'RepetitionTime not specified in metadata'
        assert len(set(trs)) == 1, 'RepetitionTime varies across runs'
        return run_files(bold_files, event_files, tr=trs[0])
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
