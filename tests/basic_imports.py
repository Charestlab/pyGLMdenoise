from unittest import TestCase, skip
import numpy


class BasicImportsTest(TestCase):

    @skip('main function not ready yet')
    def test_main_function_runs(self):
<<<<<<< HEAD
        pass
=======
        from glmdenoise.data import run_data
        out = run_data(numpy.array([]), numpy.array([]), 0, 0)
>>>>>>> master
