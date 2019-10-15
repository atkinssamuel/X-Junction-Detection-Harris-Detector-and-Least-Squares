import numpy as np
from scipy.ndimage.filters import *

def cross_junctions(I, bounds, Wpts):
    """
    Find cross-junctions in image with subpixel accuracy.

    The function locates a series of cross-junction points on a planar 
    calibration target, where the target is bounded in the image by the 
    specified quadrilateral. The number of cross-junctions identified 
    should be equal to the number of world points.

    Note also that the world and image points must be in *correspondence*,
    that is, the first world point should map to the first image point, etc.

    Parameters:
    -----------
    I       - Single-band (greyscale) image as np.array (e.g., uint8, float).
    bounds  - 2x4 np.array, bounding polygon (clockwise from upper left).
    Wpts    - 3xn np.array of world points (in 3D, on calibration target).

    Returns:
    --------
    Ipts  - 2xn np.array of cross-junctions (x, y), relative to the upper
            left corner of I. These should be floating-point values.
    """
    #--- FILL ME IN ---

    Ipts = np.zeros((2, 48))

    #------------------

    return Ipts
