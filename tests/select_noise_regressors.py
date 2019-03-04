from unittest import TestCase
import numpy


class SelectNoiseRegressorsTest(TestCase):

    def test_main_function_runs(self):
        from glmdenoise.select_noise_regressors import (
            select_noise_regressors
        )
        n = select_noise_regressors()
