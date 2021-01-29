import numpy as np


def check_inputs(data, design):
    # massage <design> and sanity-check it
    if type(design) is not list:
        design = [design]

    numcond = design[0].shape[1]
    for p in range(len(design)):
        np.testing.assert_array_equal(
            np.unique(design),
            [0, 1],
            err_msg='<design> must consist of 0s and 1s')
        condmsg = \
            'all runs in <design> should have equal number of conditions'
        np.testing.assert_equal(
            design[p].shape[1],
            numcond,
            err_msg=condmsg)
        # if the design happened to be a sparse?
        # design[p] = np.full(design[p])

    # massage <data> and sanity-check it
    if type(data) is not list:
        data = [data]

    # make sure it is single
    for p in range(len(data)):
        data[p] = data[p].astype(np.float32, copy=False)

    np.testing.assert_equal(
        np.all(np.isfinite(data[0].flatten())),
        True,
        err_msg='We checked the first run and '
        'found some non-finite values (e.g. NaN, Inf).'
        'Please fix and re-run.')
    np.testing.assert_equal(
        len(design),
        len(data),
        err_msg='<design> and <data> should have '
        'the same number of runs')

    # reshape data in 2D mode.
    is3d = data[0].ndim > 2  # is this the X Y Z T case?
    if is3d:
        xyz = data[0].shape[:3]
        n_times = data[0].shape[3]
        for p in range(len(data)):
            data[p] = np.reshape(
                data[p],
                [np.prod(xyz), n_times])
            # force to XYZ x T for convenience
    else:
        xyz = False

    return data, design, xyz
