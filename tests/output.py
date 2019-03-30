from unittest import TestCase
from unittest.mock import Mock, patch


class OutputTests(TestCase):

    @patch('glmdenoise.io.output.nibabel')
    @patch('glmdenoise.io.output.numpy')
    @patch('glmdenoise.io.output.mkdir')
    def test_non_bids_vars(self, mkdir, numpy, nibabel):
        from glmdenoise.io.output import Output
        output = Output()
        filepath='/home/johndoe/data/myproject/run_1.nii'
        output.determine_location(sample_file=filepath)
        output.save_variable(11, 'foo')
        mkdir.assert_called_with(
            '/home/johndoe/data/myproject/glmdenoise'
        )
        numpy.save.assert_called_with(
            '/home/johndoe/data/myproject/glmdenoise/foo.npy',
            11
        )
