from unittest import TestCase


class BasicImportsTest(TestCase):

    def test_main_function_runs(self):
        from glmdenoise import GLMdenoisedata
        out = GLMdenoisedata()
