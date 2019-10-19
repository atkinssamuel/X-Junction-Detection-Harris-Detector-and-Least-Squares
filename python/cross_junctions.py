import numpy as np
from scipy.ndimage.filters import *
import matplotlib.pyplot as plt


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
    m, n = I.shape

    # build the a matrix
    grid = np.indices((m, n))
    y, x = grid
    a = np.column_stack(
        (
            (x * x).reshape(-1),
            (x * y).reshape(-1),
            (y * y).reshape(-1),
            x.reshape(-1),
            y.reshape(-1),
            np.ones((m, n)).reshape(-1)
        )
    )

    b = I.reshape(-1)

    # solve the linear least squares problem defined by Eq. (4)
    p, residuals, rank, s = np.linalg.lstsq(a=a, b=b, rcond=None)
    alpha, beta, gamma, delta, epsilon, _ = p

    # compute coordinates of saddle point
    pt = - np.matmul(np.linalg.inv(np.array([[2 * alpha, beta], [beta, 2 * gamma]])), np.array([[delta], [epsilon]]))

    return pt


def distance_between_line_and_pt(line_pt1, line_pt2, pt):
    """
    Calculates the distance between a 2D line (given as two points on the line)and a 2D point.
    Assumes points are (x, y).
    :param line_pt1: one point on the line
    :param line_pt2: another point on the line
    :param pt: query point
    :return: line-point distance
    """

    return abs(
        (line_pt2[1] - line_pt1[1]) * pt[0] -
        (line_pt2[0] - line_pt1[0]) * pt[1] +
        line_pt2[0] * line_pt1[1] -
        line_pt2[1] * line_pt1[0]
    ) / distance_between_pts(line_pt1, line_pt2)


def distance_between_pts(pt1, pt2):
    """
    Calculates distance between two points.
    :param pt1: a point
    :param pt2: another point
    :return: point-point distance
    """
    return np.sqrt(((pt1 - pt2) ** 2).sum())


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
    # calculate structure tensor and harris response
    sigma = 1
    I_gaussian = gaussian_filter(I.astype(np.float32), sigma=sigma)
    I_y = sobel(I_gaussian, axis=0, mode='constant', cval=0)
    I_x = sobel(I_gaussian, axis=1, mode='constant', cval=0)

    I_x2 = gaussian_filter(I_x * I_x, sigma=sigma, mode='constant', cval=0)
    I_y2 = gaussian_filter(I_y * I_y, sigma=sigma, mode='constant', cval=0)
    I_xy = gaussian_filter(I_x * I_y, sigma=sigma, mode='constant', cval=0)

    determinant = I_x2 * I_y2 - I_xy * I_xy
    trace = I_x2 + I_y2

    response = determinant - 0.05 * trace ** 2

    # mask points outside of crude bounding box
    mask = np.ones(response.shape)
    mask[bounds[1].min():bounds[1].max() + 1, bounds[0].min():bounds[0].max() + 1] = 0
    response = np.ma.MaskedArray(response, mask=mask, fill_value=response.min())

    # mask points that are obviously not corners based on response
    response[response < 1e2] = np.ma.masked

    # find peaks by going through remaining points in descending order of response and adding them only if they
    # are far enough from previously added points
    n_desired_peaks = 80
    minimum_separation = 10
    pts = []

    for pt in np.column_stack(np.unravel_index(np.argsort(response.filled().ravel())[::-1], response.shape)):
        if len(pts) == 0:
            pts.append(pt)
        elif len(pts) == n_desired_peaks:
            break
        else:
            accepted_pts = np.row_stack(pts)
            distances = np.sqrt(((accepted_pts - pt) ** 2).sum(axis=1))
            if not np.any(distances < minimum_separation):
                pts.append(pt)
    pts = np.fliplr(np.row_stack(pts))  # now (x, y)

    # take 48 furthest points from bounding polygon
    min_distance_to_bounding_edge = [min([distance_between_line_and_pt(bounds[:, i],
                                                                       bounds[:, (i + 1) % bounds.shape[1]],
                                                                       pt)
                                          for i in range(bounds.shape[1])])
                                     for pt in pts]
    pts = pts[np.argsort(min_distance_to_bounding_edge)][-Wpts.shape[1]:]

    # sort by distance from top edge
    distance_to_top_edge = [distance_between_line_and_pt(bounds[:, 0], bounds[:, 1], pt) for pt in pts]
    pts = pts[np.argsort(distance_to_top_edge)]

    # sort by distance from left edge
    grouped_pts = np.array_split(pts, 6)
    for i, group in enumerate(grouped_pts):
        distance_to_left_edge = [distance_between_line_and_pt(bounds[:, 0], bounds[:, -1], pt) for pt in group]
        grouped_pts[i] = group[np.argsort(distance_to_left_edge)]
    pts = np.concatenate(grouped_pts)

    # use saddle point function to get sub-pixel corner locations
    Ipts = []
    w = 10
    for pt in pts:
        x, y = pt
        Ipt = saddle_point(I[y - w: y + w + 1, x - w: x + w + 1])
        Ipt[0] += x - w
        Ipt[1] += y - w
        Ipts.append(Ipt)
    Ipts = np.concatenate(Ipts, axis=1)

    return Ipts
