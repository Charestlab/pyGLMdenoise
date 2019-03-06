from unittest import TestCase, skip
from unittest.mock import Mock, patch
import numpy


class IOTest(TestCase):

    def test_run_bids(self):
        from glmdenoise.io.directory import run_bids
        bids = Mock()
        bids.get_preprocessed_subjects_ids.return_value = ['01', '02']
        tasks = {'01': ['a', 'b'], '02': ['a']}
        bids.get_tasks_for_subject.side_effect = lambda s: tasks[s]
        bids.get_filepaths_bold_runs.side_effect = lambda s, t: ('bld', s, t)
        bids.get_filepaths_event_runs.side_effect = lambda s, t: ('evt', s, t)
        bids.get_metas_bold_runs.return_value = [{'RepetitionTime': 2.2}]
        with patch('glmdenoise.io.directory.run_files') as run_files:
            run_bids(bids)
            self.assertEquals(run_files.call_count, 3)
            run_files.assert_any_call(
                ('bld', '01', 'a'), ('evt', '01', 'a'), tr=2.2)
            run_files.assert_any_call(
                ('bld', '01', 'b'), ('evt', '01', 'b'), tr=2.2)
            run_files.assert_any_call(
                ('bld', '02', 'a'), ('evt', '02', 'a'), tr=2.2)

    def test_run_bids_subject_number(self):
        from glmdenoise.io.directory import run_bids
        bids = Mock()
        tasks = {'01': ['a', 'b'], '02': ['a']}
        bids.subject_id_from_number.side_effect = lambda sn: '0' + str(sn)
        bids.get_tasks_for_subject.side_effect = lambda s: tasks[s]
        bids.get_filepaths_bold_runs.side_effect = lambda s, t: ('bld', s, t)
        bids.get_filepaths_event_runs.side_effect = lambda s, t: ('evt', s, t)
        bids.get_metas_bold_runs.return_value = [{'RepetitionTime': 2.2}]
        with patch('glmdenoise.io.directory.run_files') as run_files:
            run_bids(bids, sub_num=1)
            self.assertEquals(run_files.call_count, 2)
            run_files.assert_any_call(
                ('bld', '01', 'a'), ('evt', '01', 'a'), tr=2.2)
            run_files.assert_any_call(
                ('bld', '01', 'b'), ('evt', '01', 'b'), tr=2.2)

    def test_run_bids_subject_task(self):
        from glmdenoise.io.directory import run_bids
        bids = Mock()
        bids.get_filepaths_bold_runs.side_effect = lambda s, t: ('bld', s, t)
        bids.get_filepaths_event_runs.side_effect = lambda s, t: ('evt', s, t)
        bids.get_metas_bold_runs.return_value = [{'RepetitionTime': 2.2}]
        with patch('glmdenoise.io.directory.run_files') as run_files:
            run_bids(bids, sub='01', task='a')
            self.assertEquals(run_files.call_count, 1)
            run_files.assert_called_with(
                ('bld', '01', 'a'), ('evt', '01', 'a'), tr=2.2)

    def test_run_bids_subject_separate_sessions(self):
        from glmdenoise.io.directory import run_bids
        bids = Mock()
        bids.get_sessions_for_task_and_subject.return_value = ['1', '2']
        bids.get_filepaths_bold_runs.side_effect = lambda s, t: ('bld', s, t)
        bids.get_filepaths_event_runs.side_effect = lambda s, t: ('evt', s, t)
        bids.get_metas_bold_runs.return_value = [{'RepetitionTime': 2.2}]
        with patch('glmdenoise.io.directory.run_files') as run_files:
            run_bids(bids, sub='01', task='a')
            self.assertEquals(run_files.call_count, 1)
            run_files.assert_called_with(
                ('bld', '01', 'a'), ('evt', '01', 'a'), tr=2.2)
