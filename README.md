# pyGLMdenoise
python implementation of GLMdenoise

[full documentation](http://glmdenoise.readthedocs.io/)


### Usage

##### Python

```python
from glmdenoise import GLMdenoisedata

GLMdenoisedata(design, data, stimdur=0.5, tr=2)
```

##### Console

```sh
glmdenoise /my/data/dir
```

### Installation

```sh
pip install glmdenoise
```

### Unit tests

To run the unit tests:

```sh
python setup.py tests -q
```
