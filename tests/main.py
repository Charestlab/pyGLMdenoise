from unittest import TestCase, skipIf
from numpy.random import RandomState
import pandas
import socket


class MainClassTest(TestCase):

    @skipIf(
        socket.gethostname() == 'colles-d1800479',
        'RandomState different on Jaspers workstation'
    )
    def test_fit_assume_hrf(self):
        from glmdenoise.pyGlmdenoise import GLMdenoise
        rng = RandomState(seed=156336647)
        design1 = pandas.DataFrame([
            {'onset': 0, 'duration': 0.5, 'trial_type': 'foo'},
            {'onset': 1, 'duration': 0.5, 'trial_type': 'bar'},
            {'onset': 2, 'duration': 0.5, 'trial_type': 'foo'},
            {'onset': 3, 'duration': 0.5, 'trial_type': 'bar'}
        ])
        design = [design1] * 3
        data = [
            rng.rand(4, 5),
            rng.rand(4, 5),
            rng.rand(4, 5)
        ]
        gd = GLMdenoise(params={'hrfmodel': 'assume'})
        gd.fit(design, data, 1.0)
        self.assertEqual(gd.results['select_pca'], 1)
