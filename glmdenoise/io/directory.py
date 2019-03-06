from glmdenoise.io.files import run_files
from glmdenoise.io.bids import BidsDirectory


def run_bids_directory(directory='.', sub_num=None, sub=None, task=None):
    bids = BidsDirectory(directory)
    return run_bids(bids, sub_num=sub_num, sub=sub, task=task)


def run_bids(bids, sub_num=None, sub=None, task=None):
    if sub and task:
        bold_files = bids.get_filepaths_bold_runs(sub, task)
        event_files = bids.get_filepaths_event_runs(sub, task)
        metas = bids.get_metas_bold_runs(sub, task)
        trs = [meta['RepetitionTime'] for meta in metas]
        assert trs, 'RepetitionTime not specified in metadata'
        assert len(set(trs)) == 1, 'RepetitionTime varies across runs'
        run_files(bold_files, event_files, tr=trs[0])

