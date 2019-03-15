import numpy as np


def isrowvector(m):
    if not isinstance(m, np.ndarray):
        m = np.asarray(m)
    f = m.shape[0] == 1
    return f