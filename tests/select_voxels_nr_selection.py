from unittest import TestCase
from numpy.testing import assert_array_equal
import numpy


class VoxelsNoiseRegressorSelectionTest(TestCase):

    def test_select_without_mask(self):
        from glmdenoise.select_voxels_nr_selection import (
            select_voxels_nr_selection
        )
        # rows are voxels, columns nr solutions
        r2_voxels_nrs = numpy.array([
            [ 1,  1],
            [-1,  1],
            [ 1,  0],
            [-1, -1],
            [ 0,  0],
        ])
        indices = select_voxels_nr_selection(r2_voxels_nrs)
        assert_array_equal(indices,
            numpy.array([True, True, True, False, False]))
