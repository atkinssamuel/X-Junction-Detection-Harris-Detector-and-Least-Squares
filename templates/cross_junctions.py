import numpy as np
from scipy.ndimage.filters import *
from matplotlib.path import Path

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
    # --- FILL ME IN ---
    m, n = I.shape
    # Loop over entire image for each x and y point:
    A = np.zeros([m * n, 6])
    q = np.zeros([m * n, 1])
    row_index = 0
    for y in range(m):
        for x in range(n):
            A[row_index] = np.array([x * x, x * y, y * y, x, y, 1])
            q[row_index] = I[y, x]
            row_index += 1
    # Now that the matrices are configured, we can use least squares:
    [alpha], [beta], [gamma], [delta], [epsilon], [zeta] \
        = np.linalg.lstsq(np.dot(A.T, A), np.dot(A.T, q), rcond=-1)[0]

    # ------------------
    C = np.array([[2 * alpha, beta], [beta, 2 * gamma]])
    D = np.array([[delta], [epsilon]])
    try:
        pt = -np.dot(np.linalg.inv(C), D)
    except:
        print(I.shape)
        return np.array([I.shape[0]/2, I.shape[1]/2]).reshape(2, 1)
    if pt[0] < 0 or pt[0] > I.shape[1] or pt[1] < 0 or pt[1] > I.shape[0]:
        print("Out of range")
        return np.array([I.shape[0]/2, I.shape[1]/2]).reshape(2, 1)
    return pt


def harris_corner_detector(I, checkerboard_border):
    gaussian_blur = np.array([[1, 1, 1, 1, 1],
                              [1, 1, 1, 1, 1],
                              [1, 1, 1, 1, 1],
                              [1, 1, 1, 1, 1],
                              [1, 1, 1, 1, 1]])
    I = convolve(I, gaussian_blur)
    x_derivative_kernel = np.array([[1, 0, -1],
                                    [1, 0, -1],
                                    [1, 0, -1]])/9
    y_derivative_kernel = np.array([[1, 1, 1],
                                    [0, 0, 0],
                                    [-1, -1, -1]])/9
    i_x = convolve(I, x_derivative_kernel)
    i_y = convolve(I, y_derivative_kernel)

    corner_array = []

    extracted_points = []
    max_value = 0
    R = np.zeros([I.shape[0], I.shape[1]])
    for y in range(I.shape[0]):
        for x in range(I.shape[1]):
            if checkerboard_border is None or checkerboard_border.contains_point([x, y]):
                A = np.array([[i_x[y, x] * i_x[y, x], i_x[y, x] * i_y[y, x]],
                              [i_x[y, x] * i_y[y, x], i_y[y, x] * i_y[y, x]]])
                A = convolve(A, gaussian_filter(A, 1))
                scale = 0.10
                R_i = np.linalg.det(A) - scale * np.trace(A) ** 2
                R[y, x] = R_i
                if R_i > max_value:
                    max_value = R_i

    for y in range(I.shape[0]):
        for x in range(I.shape[1]):
            if checkerboard_border is None or checkerboard_border.contains_point([x, y]):
                if R[y, x] > 0.1 * max_value:
                        #and R[y, x] > R[y - 1, x - 1] and R[y, x] > R[y, x - 1] \
                        #and R[y, x] > R[y + 1, x - 1] and R[y, x] > R[y - 1, x] and R[y, x] > R[y + 1, x] \
                        #and R[y, x] > R[y - 1, x + 1] and R[y, x] > R[y, x + 1] and R[y, x] > R[y + 1, x + 1]:
                    extracted_points.append([x, y, R[y, x]])

    extracted_points = np.array(extracted_points)
    extracted_points = extracted_points[extracted_points[:, 2].argsort()]
    extracted_points = extracted_points[:1]
    return extracted_points


def cross_junctions(I, bounds, Wpts, scale=0.15):
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
    # --- FILL ME IN ---
    m, n = I.shape
    bounds = bounds.T
    UL = np.array([bounds[0][0], bounds[0][1]])
    UR = np.array([bounds[1][0], bounds[1][1]])
    BR = np.array([bounds[2][0], bounds[2][1]])
    BL = np.array([bounds[3][0], bounds[3][1]])
    bounds = bounds.T

    # "New" = N points
    NUL = UL + (UR - UL) * scale + (BL - UL) * scale
    NUR = UR + (UL - UR) * scale + (BR - UR) * scale
    NBR = BR + (BL - BR) * scale + (UR - BR) * scale
    NBL = BL + (BR - BL) * scale + (UL - BL) * scale

    bounds_list = NUL, NUR, NBR, NBL

    Ipts = []

    bpoly = [NUL, NUR, NBR, NBL, NUL]
    codes = [
        Path.MOVETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.CLOSEPOLY
    ]
    corner_border = Path(bpoly, codes)

    left_points = []
    right_points = []

    left_down_vector = NBL - NUL
    left_vector_increment = left_down_vector/5
    right_down_vector = NBR - NUR
    right_vector_increment = right_down_vector/5
    refined_points = []
    for row in range(6):
        right_points.append(NUR + right_vector_increment * row)
        left_points.append(NUL + left_vector_increment * row)
    unrefined_points = []
    for y in range(6):
        right_vector = right_points[y] - left_points[y]
        increment = right_vector/7
        for x in range(8):
            point = NUL + left_vector_increment * y + increment * x
            unrefined_points.append([point[0], point[1]])
            window_length = np.linalg.norm(right_vector)/20
            image_slice = I[int(np.floor(point[1] - window_length)) - 1: int(np.ceil(point[1] + window_length)) + 1,
                          int(np.floor(point[0] - window_length)) - 1: int(np.ceil(point[0] + window_length)) + 1]
            refined_point = saddle_point(image_slice)
            refined_point = [refined_point[0] + int(point[0] - window_length) - 1,
                             refined_point[1] + int(point[1] - window_length) - 1]
            refined_points.append([refined_point[0], refined_point[1]])
    refined_points = np.array(refined_points)
    unrefined_points = np.array(unrefined_points)
    try:
        refined_points = refined_points.squeeze(2)
        unrefined_points = unrefined_points.squeeze(2)
    except:
        pass
    # ------------------
    return refined_points.T
