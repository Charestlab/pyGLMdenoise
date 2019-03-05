from unittest import TestCase, skip
import numpy


class BasicImportsTest(TestCase):

    @skip('main function not ready yet')
    def test_main_function_runs(self):
        from glmdenoise import GLMdenoisedata
        out = GLMdenoisedata(numpy.array([]), numpy.array([]), 0, 0)
