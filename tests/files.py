from unittest import TestCase, skip
from unittest.mock import Mock, patch


class FilesTests(TestCase):

    @patch('glmdenoise.io.files.nibabel')
    @patch('glmdenoise.io.files.pandas')
    @patch('glmdenoise.io.files.GLMdenoise')
    @patch('glmdenoise.io.files.Output')
    def test_run_files_plots_figs(self, Output, GLMdenoise, pandas, nibabel):
        from glmdenoise.io.files import run_files
        glmdenoise = GLMdenoise.return_value
        out = Output.return_value
        bids = Mock()
        data1, design1 = Mock(), Mock()
        run_files([data1], [design1], 1.0)
        Output.assert_called_with(data1, None)
        glmdenoise.plot_figures.assert_called_with(out.create_report())

    @patch('glmdenoise.io.files.nibabel')
    @patch('glmdenoise.io.files.pandas')
    @patch('glmdenoise.io.files.GLMdenoise')
    @patch('glmdenoise.io.files.Output')
    def test_run_files_saves_data(self, Output, GLMdenoise, pandas, nibabel):
        from glmdenoise.io.files import run_files
        glmdenoise = GLMdenoise.return_value
        glmdenoise.results.get.side_effect = lambda k: '$' + k
        out = Output.return_value
        bids = Mock()
        data1, design1 = Mock(), Mock()
        run_files([data1], [design1], 1.0)
        Output.assert_called_with(data1, None)
        out.save_image.assert_called_with('$pseudo_t_stats', 'pseudo_t_stats')
        out.save_variable.assert_called_with('$xval', 'xval')
