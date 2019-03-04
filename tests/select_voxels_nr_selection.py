from unittest import TestCase
import numpy


class VoxelsNoiseRegressorSelectionTest(TestCase):

    def test_main_function_runs(self):
        from glmdenoise.select_voxels_nr_selection import (
            select_voxels_nr_selection
        )
        indices = select_voxels_nr_selection()
