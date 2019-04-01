from unittest import TestCase
from unittest.mock import Mock, patch


class OutputTests(TestCase):

    @patch('glmdenoise.io.output.nibabel')
    @patch('glmdenoise.io.output.numpy')
    @patch('glmdenoise.io.output.mkdir')
    def test_ensures_directory(self, mkdir, _np, _nb):
        from glmdenoise.io.output import Output
        output = Output()
        filepath='/home/johndoe/data/myproject/run_1.nii'
        output.configure_from(sample_file=filepath)
        output.save_variable(11, 'foo')
        mkdir.assert_called_with(
            '/home/johndoe/data/myproject/glmdenoise'
        )

    @patch('glmdenoise.io.output.nibabel')
    def test_file_path_non_bids(self, nibabel):
        from glmdenoise.io.output import Output
        output = Output()
        filepath='/home/johndoe/data/myproject/run_1.nii'
        output.configure_from(sample_file=filepath)
        self.assertEqual(
            output.file_path('bar', 'xyz'),
            '/home/johndoe/data/myproject/glmdenoise/bar.xyz'
        )

    def test_file_path_bids(self):
        from glmdenoise.io.output import Output
        output = Output()
        bids = Mock()
        bids.root = '/d'
        output.fit_bids_context(bids, sub='1', ses='2', task='a')
        self.assertEqual(
            output.file_path('bar', 'xyz'),
            '/d/derivatives/glmdenoise/sub-1/ses-2/sub-1_ses-2_task-a_bar.xyz'
        )
