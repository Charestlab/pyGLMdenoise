from unittest import TestCase, skip
from unittest.mock import Mock, patch


class FilesTests(TestCase):

    @patch('glmdenoise.io.files.nibabel')
    @patch('glmdenoise.io.files.pandas')
    @patch('glmdenoise.io.files.GLMdenoise')
    @patch('glmdenoise.io.files.Output')
    def test_run_files(self, Output, GLMdenoise, pandas, nibabel):
        from glmdenoise.io.files import run_files
        glmdenoise = GLMdenoise.return_value
        output = Output.return_value
        bids = Mock()
        data1, design1 = Mock(), Mock()
        run_files([data1], [design1], 1.0)
        Output.assert_called_with(data1, None)
        glmdenoise.plot_figures.assert_called_with(output.create_report())
        
