from glmdenoise.pyGlmdenoise import GLMdenoise
import numpy.random
import pandas

rng = numpy.random.RandomState(seed=156336647)

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
