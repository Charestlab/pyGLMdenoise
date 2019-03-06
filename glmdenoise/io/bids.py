from bids import BIDSLayout


class BidsDirectory(object):

    def __init__(self, directory):
        pass

    def get_preprocessed_subjects_ids(self):
        pass
    
    def get_tasks_for_subject(self, subject):
        pass

    def get_filepaths_bold_runs(self, subject, task):
        pass

    def get_filepaths_event_runs(self, subject, task):
        pass

    def get_metas_bold_runs(self, subject, task):
        pass

    def subject_id_from_number(self, sub_num):
        pass
