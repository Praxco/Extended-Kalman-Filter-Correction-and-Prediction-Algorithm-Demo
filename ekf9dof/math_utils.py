"""Small, dependency-light math helpers shared across the filter."""

import numpy as np


def skew(v):
    """Return the 3x3 skew-symmetric (cross-product) matrix of a 3-vector.

    skew(a) @ b == np.cross(a, b)
    """
    x, y, z = v
    return np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ]
    )
