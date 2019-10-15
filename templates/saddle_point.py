from builtins import range

import numpy as np
from numpy.linalg import inv, lstsq

def saddle_point(I):
    """
    Locate saddle point in an image patch.

    The function identifies the subpixel centre of a cross-junction in the
    image patch I, by fitting a hyperbolic paraboloid to the patch, and then 
    finding the critical point of that paraboloid.

    Note that the location of 'p' is relative to (-0.5, -0.5) at the upper
    left corner of the patch, i.e., the pixels are treated as covering an 
    area of one unit square.

    Parameters:
    -----------
    I  - Single-band (greyscale) image patch as np.array (e.g., uint8, float).

    Returns:
    --------
    pt  - 2x1 np.array, subpixel location of saddle point in I (x, y coords).
    """
    #--- FILL ME IN ---

    m, n = I.shape
    # Loop over entire image for each x and y point:
    A = np.zeros([m*n, 6])
    q = np.zeros([m*n, 1])
    row_index = 0
    for y in range(m):
        for x in range(n):
            A[row_index] = np.array([x*x, x*y, y*y, x, y, 1])
            q[row_index] = I[y, x]
            row_index += 1
    print(A.shape)
    # Now that the matrices are configured, we can use least squares:
    [alpha], [beta], [gamma], [delta], [epsilon], [zeta] \
        = np.linalg.lstsq(np.dot(A.T, A), np.dot(A.T, q), rcond=None)[0]

    #------------------
    C = np.array([[2 * alpha, beta], [beta, 2 * gamma]])
    D = np.array([[delta], [epsilon]])
    pt = -np.dot(inv(C), D)
    return pt
