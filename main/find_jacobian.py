import numpy as np

def find_jacobian(K, Twc, Wpt):
    """
    Determine the Jacobian for NLS camera pose optimization.

    The function computes the Jacobian of an image plane point with respect
    to the current camera pose estimate, given a landmark point. The 
    projection model is the simple pinhole model.

    Parameters:
    -----------
    K    - 3x3 np.array, camera intrinsic calibration matrix.
    Twc  - 4x4 np.array, homogenous pose matrix, current guess for camera pose. 
    Wpt  - 3x1 world point on calibration target (one of n).

    Returns:
    --------
    J  - 2x6 np.array, Jacobian matrix (columns are tx, ty, tz, r, p, q).
    """
    #--- FILL ME IN ---

    # Extracting rotation matrix from provided Twc:
    rotation_matrix = Twc[:3, :3]

    # Extracting transform (last column) from Twc:
    transform = Twc[:3, -1:]

    # dt, f:
    dt = Wpt - transform
    f = K.dot(rotation_matrix.T).dot(dt)

    # df:
    df = np.zeros([3, 6])
    df[:, :3] = K.dot(rotation_matrix.T).dot((-np.eye(3)))

    c, s = rotation_matrix[:2, 0] / np.sqrt(1 - rotation_matrix[2, 0] * rotation_matrix[2, 0])

    # defining dr, dy, and dp matrices:
    dr = rotation_matrix.dot(np.array([[0, 0, 0],
                                       [0, 0, -1],
                                       [0, 1, 0]]))
    dy = np.array([[0, -1, 0],
                  [1, 0, 0],
                  [0, 0, 0]]).dot(rotation_matrix)
    dp = np.array([[0, 0, c],
                   [0, 0, s],
                   [-c, -s, 0]]).dot(rotation_matrix)

    df[:, 3:4] = K.dot(dr.T).dot(dt)
    df[:, 4:5] = K.dot(dp.T).dot(dt)
    df[:, 5:6] = K.dot(dy.T).dot(dt)

    dfz = df[-1:, :]

    # Applying Jacobian formula:
    J = np.divide((f[-1, 0] * df - f.dot(dfz)), f[-1, 0] * f[-1, 0])[:-1, :]

    #------------------

    return J
