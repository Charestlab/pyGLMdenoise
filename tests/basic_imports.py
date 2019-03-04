from unittest import TestCase
import numpy


class BasicImportsTest(TestCase):

    def test_main_function_runs(self):
        from glmdenoise import GLMdenoisedata
        out = GLMdenoisedata(numpy.array([]), numpy.array([]), 0, 0)
