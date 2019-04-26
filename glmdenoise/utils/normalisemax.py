
import numpy as np
from glmdenoise.utils.choose import choose as ch
from glmdenoise.utils.isrowvector import isrowvector as isr


def normalisemax(m, dim=None):
    """Divide array by the max value along some dimension

    f = normalisemax(m,dim)

    <m> is a matrix
    <dim> (optional) is the dimension of <m> to operate upon.
    default to 2 if <m> is a row vector and to 1 otherwise.
    special case is 0 which means operate globally.

    divide <m> by the max value along some dimension (or globally).

    example:
    (normalisemax([[1, 2, 3]])==[[1/3, 2/3, 1]]).all()
    """

    # input
    if dim is None:
        dim = ch(isr(m), 1, 0)
    # do it
    if dim == 0:
        f = m / np.max(m)
    else:
        f = m / np.max(m, dim)
    return f
