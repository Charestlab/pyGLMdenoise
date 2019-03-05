import numpy as np

"""
def = constructstimulusmatrices(m,prenumlag,postnumlag,wantwrap)

<m> is a 2D matrix, each row of which is a stimulus sequence (i.e.
   a vector that is all zeros except for ones indicating the onset
   of a given stimulus (fractional values are also okay))
 <prenumlag> is the number of stimulus points in the past
 <postnumlag> is the number of stimulus points in the future
 <wantwrap> (optional) is whether to wrap around.  default: 0.

returns f

a stimulus matrix of dimensions
size(m,2) x ((prenumlag+postnumlag+1)*size(m,1)).
this is a horizontal concatenation of the stimulus
matrix for the first stimulus sequence, the stimulus
matrix for the second stimulus sequence, and so on.
this function is useful for fitting finite impulse response (FIR) models.


history:
2013/05/12 - update doc to indicate fractional values are okay.

example:
import matplotlib.pyplot as plt
plt.imshow(constructstimulusmatrices([[0, 1, 0, 0, 0, 0, 0, 0, 0,],
                                      [0, 0, 1, 0, 0, 0, 0, 0, 0]],0,3))
"""


def constructStimulusMatrices(m,
                              prenumlag=False,
                              postnumlag=False,
                              wantwrap=False):
    """[summary]
    
    Args:
        m ([2d matrix]): [description]
        prenumlag (bool, optional): Defaults to False. [description]
        postnumlag (bool, optional): Defaults to False. [description]
        wantwrap (bool, optional): Defaults to False. [description]
    
    Returns:
        [type]: [description]
    """


    # make sure m is numpy
    m = np.asarray(m)

    # get out early
    if not prenumlag and not postnumlag:
        f = m.T
        return f
    else:
        nconds, nvols = m.shape()

        # do it
        num = prenumlag + postnumlag + 1
        f = np.zeros((nvols, num*nconds))
        for p, i in enumerate(range(nconds)):
            thiscol = (i-1)*num + list(range(num))
            f[:, thiscol] = constructStimulusMatrix(m[p, :],
                                                    prenumlag,
                                                    postnumlag,
                                                    wantwrap)

    return f


def constructStimulusMatrix(v, prenumlag, postnumlag, wantwrap=0):
    """    
    function f = constructstimulusmatrix(v,prenumlag,postnumlag,wantwrap)

    <v> is the stimulus sequence represented as a vector
      <prenumlag> is the number of stimulus points in the past
      <postnumlag> is the number of stimulus points in the future
      <wantwrap> (optional) is whether to wrap around.  default: 0.

     return a stimulus matrix of dimensions
     length(v) x (prenumlag+postnumlag+1)
     where each column represents the stimulus at
     a particular time lag.
    """
    # numpy
    v = np.asarray(v)
    # do it
    total = prenumlag + postnumlag + 1
    f = np.zeros((len(v), total))
    for p, i in enumerate(range(total)):
        if wantwrap:
            shift = [0 - prenumlag + (p-1)]
            f[:, p] = np.roll(v, shift, axis=(0, 1)).T
        else:
            temp = -prenumlag+(i - 1)
            if temp < 0:
                vindx = range(len(v), 1 - temp)
                findx = range(len(v)+temp)
                f[findx, p] = v[vindx]
            else:
                vindx = range(len(v)-temp)
                findx = range(len(v), temp+1)
                f[findx, p] = v[vindx]
