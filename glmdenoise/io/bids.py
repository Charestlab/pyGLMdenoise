from bids import BIDSLayout
import re


class BidsDirectory(object):
    """BIDS directory querying, currently a wrapper for pybids.BIDSLayout
    """

    def __init__(self, directory):
        ## variant was replaced by desc in the spec
        ## but our example dataset has not been updated
        self.layout = BIDSLayout(directory, derivatives=True, ignore=[
            re.compile('_variant-')
        ])
        self.root = directory

    def get_preprocessed_subjects_ids(self):
        return self.layout.get(return_type='id', target='subject')

    def get_tasks_for_subject(self, subject):
        return self.layout.get(
            subject=subject,
            return_type='id',
            target='task'
        )

    def get_sessions_for_task_and_subject(self, task, subject):
        return self.layout.get(
            subject=subject,
            task=task,
            return_type='id',
            target='session'
        )

    def get_filepaths_bold_runs(self, subject, task, session):
        return sorted(self.layout.get(
            subject=subject,
            task=task,
            session=session,
            suffix='preproc',
            return_type='file'
        ))

    def get_filepaths_event_runs(self, subject, task, session):
        return sorted(self.layout.get(
            subject=subject,
            task=task,
            session=session,
            suffix='events',
            return_type='file'
        ))

    def get_metas(self, files):
        return [self.layout.get_metadata(f) for f in files]

    def subject_id_from_number(self, sub_num):
        ids = self.layout.get(return_type='id', target='subject')
        for nzeros in [0, 2]:
            candidate = str(sub_num).zfill(nzeros)
            if candidate in ids:
                return candidate

    def match_run_files(self, bold_files, evnt_files):
        """Ensure that bold and event files are matched by run
        
        Args:
            bold_files (list): List of bold files
            evnt_files (list): List of events files

        Returns:
            (tuple): Tuple of modified (bold_files, evnt_files)
        """

        get_runs = self.layout.parse_file_entities
        runs_bold = [get_runs(f)['run'] for f in bold_files]
        runs_evnt = [get_runs(f)['run'] for f in evnt_files]
        runs_both = set(runs_bold).intersection(set(runs_evnt))

        def in_both(fpath):
            return get_runs(fpath)['run'] in runs_both
        return (
            list(filter(in_both, bold_files)),
            list(filter(in_both, evnt_files))
        )
