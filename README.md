# pyGLMdenoise
python implementation of GLMdenoise

[full documentation](http://glmdenoise.readthedocs.io/)


### Installation

```sh
pip install glmdenoise
```

### Usage

##### Python

```python
from glmdenoise import GLMdenoise
gd = GLMdenoise()
gd.fit(design, data, tr=2.0)
gd.plot_figures()
```

##### Console

```sh
glmdenoise /my/data/dir
```

##### Public dataset

```sh
glmdenoise ///workshops/nih-2017/ds000114
```

### Unit tests

To run the unit tests:

```sh
python setup.py tests -q
```
