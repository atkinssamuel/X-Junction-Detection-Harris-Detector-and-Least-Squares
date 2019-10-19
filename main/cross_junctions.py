import numpy as np
from scipy.ndimage.filters import *
from matplotlib.path import Path
import matplotlib.pyplot as plt

def sort_points(points, bounds, Wpts):
    # Now we have to filter the points to make the points match up with the Wpts:
    bounding_edge_displacement = get_bounding_edge_displacement(points, bounds)
    sorting_index = np.argsort(bounding_edge_displacement)
    points = points[sorting_index][-Wpts.shape[1]:]

    # Distance from UL-UR edge:
    top_edge_displacement = [line_point_displacement(bounds[:, 0], bounds[:, 1], point) for point in points]
    points = points[np.argsort(top_edge_displacement)]

    split_points = np.array_split(points, 6)
    point_array = []
    for i, group in enumerate(split_points):
        # Distance from UL-BL edge:
        left_edge_displacement = [line_point_displacement(bounds[:, 0], bounds[:, -1], pt) for pt in group]
        # Grouping and sorting points based on their displacement from the left edge:
        split_points[i] = group[np.argsort(left_edge_displacement)]
        for point in split_points[i]:
            point_array.append(point)
    return point_array

def point_displacement(A, B):
    # Returns the distance between point A and point B:
    return np.sqrt(((A - B) ** 2).sum())


def line_point_displacement(line_pt1, line_pt2, pt):
    return abs(
        (line_pt2[1] - line_pt1[1]) * pt[0] -
        (line_pt2[0] - line_pt1[0]) * pt[1] +
        line_pt2[0] * line_pt1[1] -
        line_pt2[1] * line_pt1[0]
    ) / point_displacement(line_pt1, line_pt2)


def get_bounding_edge_displacement(points, bounds):
    m, n = bounds.shape
    return [min([line_point_displacement(bounds[:, index], bounds[:, (index + 1) % n],
                                         point)
                 for index in range(n)])
            for point in points]


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
        return np.array([I.shape[0] / 2, I.shape[1] / 2]).reshape(2, 1)
    if pt[0] < 0 or pt[0] > I.shape[1] or pt[1] < 0 or pt[1] > I.shape[0]:
        print("Out of range")
        return np.array([[I.shape[0] / 2], [I.shape[1] / 2]])
    return pt


def harris_corner_detector(I, corner_border, sigma):
    # Passing Gaussian over image to reduce false corner detection:
    I_gauss = gaussian_filter(I.astype(np.float32), sigma=sigma)
    I_y = sobel(I_gauss, axis=0, mode='constant', cval=0)
    I_x = sobel(I_gauss, axis=1, mode='constant', cval=0)

    Ixx = gaussian_filter(I_x * I_x, sigma=1, mode='constant', cval=0)
    Ixy = gaussian_filter(I_x * I_y, sigma=1, mode='constant', cval=0)
    Iyy = gaussian_filter(I_y * I_y, sigma=1, mode='constant', cval=0)

    d = Ixx * Iyy - Ixy * Ixy
    t = Ixx + Iyy

    R = d - 0.05 * t ** 2

    for y in range(R.shape[0]):
        for x in range(R.shape[1]):
            if not corner_border.contains_point([x, y]):
                R[y, x] = 0

    R = np.ma.MaskedArray(R, fill_value=R.min())

    # Parameters:
    point_count = 48
    min_displacement = 20
    accepted_points = []

    # Arranging pts_column:
    sorted_points = np.argsort(R.filled().ravel())
    pts_column = np.unravel_index(sorted_points[::-1], R.shape)
    pts_column = np.column_stack(pts_column)

    # Iterating through each point:
    for prospective_point in pts_column:
        if len(accepted_points) == point_count:
            break
        elif len(accepted_points) == 0:
            accepted_points.append(prospective_point)
        else:
            accepted_pts = np.row_stack(accepted_points)
            distances = np.sqrt(((accepted_pts - prospective_point) ** 2))
            distances = distances.sum(axis=1)
            if np.any(distances < min_displacement):
                continue
            else:
                accepted_points.append(prospective_point)

    return np.fliplr(np.row_stack(accepted_points))


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
    # --- FILL ME IN ---
    m, n = I.shape
    bounds = bounds.T
    UL = np.array([bounds[0][0], bounds[0][1]])
    UR = np.array([bounds[1][0], bounds[1][1]])
    BR = np.array([bounds[2][0], bounds[2][1]])
    BL = np.array([bounds[3][0], bounds[3][1]])
    bottom_scale = 0.09
    top_scale = 0.12
    left_scale = 0.09
    right_scale = 0.08

    bounds = bounds.T

    # "New" = N points
    NUL = UL + (UR - UL) * left_scale + (BL - UL) * top_scale
    NUR = UR + (UL - UR) * right_scale + (BR - UR) * top_scale
    NBR = BR + (BL - BR) * right_scale + (UR - BR) * bottom_scale
    NBL = BL + (BR - BL) * left_scale + (UL - BL) * bottom_scale

    bounds_list = NUL, NUR, NBR, NBL
    outer_bounds_list = UL, UR, BR, BL

    inner_bounding_poly = [NUL, NUR, NBR, NBL, NUL]
    outer_bounding_poly = [UL, UR, BR, BL, UL]
    codes = [
        Path.MOVETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.CLOSEPOLY
    ]
    inner_border = Path(inner_bounding_poly, codes)
    outer_border = Path(outer_bounding_poly, codes)

    sigma = 11
    points = harris_corner_detector(I, inner_border, sigma)

    # Computing the exact point using the saddle_point function defined above:
    updated_points = []
    window = 15
    for point in points:
        up = point[1] - window
        down = point[1] + window
        left = point[0] - window
        right = point[0] + window
        if right > I.shape[1]:
            print("Too right")
            right = I.shape[1]
        if left < 0:
            print("Too left")
            left = 0
        if up < 0:
            print("Too up")
            up = 0
        if down > I.shape[0]:
            print("Too down")
            down = I.shape[0]
        image_splice = I[up: down, left: right]
        updated_point = saddle_point(image_splice)
        updated_point = np.array([updated_point[0] + left, updated_point[1] + up])
        updated_points.append([updated_point[0], updated_point[1]])
    updated_points = np.array(updated_points).reshape(48, 2)
    sorted_points = sort_points(updated_points, bounds, Wpts)
    return np.array(sorted_points).T, outer_border
