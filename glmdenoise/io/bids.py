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

    def get_metas_bold_runs(self, subject, task, session):
        runs = self.layout.get(
            subject=subject,
            task=task,
            session=session,
            suffix='bold'  # get metadata from raw!?
        )
        return [r.metadata for r in sorted(runs, key=lambda b: b.filename)]

    def subject_id_from_number(self, sub_num):
        ids = self.layout.get(return_type='id', target='subject')
        for nzeros in [0, 2]:
            candidate = str(sub_num).zfill(nzeros)
            if candidate in ids:
                return candidate

    def match_run_files(self, bold_files, evnt_files):
        get_runs = self.layout.parse_file_entities
        runs_bold = [get_runs(f)['run'] for f in bold_files]
        runs_evnt = [get_runs(f)['run'] for f in evnt_files]
        runs_both = set(runs_bold).intersection(set(runs_evnt))
        for f in range(len(bold_files)):
            if runs_bold[f] not in runs_both:
                del bold_files[f]
        for f in range(len(evnt_files)):
            if runs_evnt[f] not in runs_both:
                del evnt_files[f]

        
        
        
