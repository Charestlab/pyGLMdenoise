from bids import BIDSLayout


class BidsDirectory(object):
    """BIDS directory querying, currently a wrapper for pybids.BIDSLayout
    """


    def __init__(self, directory):
        self.layout = BIDSLayout(directory, derivatives=True)

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
        return self.layout.get(
            subject=subject,
            task=task,
            session=session,
            suffix='preproc',
            return_type='file'
        )

    def get_filepaths_event_runs(self, subject, task, session):
        return self.layout.get(
            subject=subject,
            task=task,
            session=session,
            suffix='events',
            return_type='file'
        )

    def get_metas_bold_runs(self, subject, task, session):
        runs = self.layout.get(
            subject=subject,
            task=task,
            session=session,
            suffix='bold'  # get metadata from raw!?
        )
        return [r.metadata for r in runs]

    def subject_id_from_number(self, sub_num):
        ids = self.layout.get(return_type='id', target='subject')
        for nzeros in [0, 2]:
            candidate = str(sub_num).zfill(nzeros)
            if candidate in ids:
                return candidate
