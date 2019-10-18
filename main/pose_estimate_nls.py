import numpy as np
from numpy.linalg import inv, norm
from find_jacobian import find_jacobian
from dcm_from_rpy import dcm_from_rpy
from rpy_from_dcm import rpy_from_dcm

def pose_estimate_nls(K, Twcg, Ipts, Wpts):
    """
    Estimate camera pose from 2D-3D correspondences via NLS.

    The function performs a nonlinear least squares optimization procedure 
    to determine the best estimate of the camera pose in the calibration
    target frame, given 2D-3D point correspondences.

    Parameters:
    -----------
    Twcg  - 4x4 homogenous pose matrix, initial guess for camera pose.
    Ipts  - 2xn array of cross-junction points (with subpixel accuracy).
    Wpts  - 3xn array of world points (one-to-one correspondence with Ipts).

    Returns:
    --------
    Twc  - 4x4 np.array, homogenous pose matrix, camera pose in target frame.
    """
    maxIters = 250                          # Set maximum iterations.

    tp = Ipts.shape[1]                      # Num points.
    ps = np.reshape(Ipts, (2*tp, 1), 'F')   # Stacked vector of observations.

    dY = np.zeros((2*tp, 1))                # Residuals.
    J = np.zeros((2*tp, 6))                # Jacobian.

    #--- FILL ME IN ---
    E_pose = epose_from_hpose(Twcg)
 
    for i in range(maxIters):
        H_pose = hpose_from_epose(E_pose)

        # Constructing A and b matrices to house Jacobian:
        A = np.zeros([6, 6])
        b = np.zeros([6, 1])
        for j in range(Ipts.shape[1]):
            J = find_jacobian(K, H_pose, Wpts[:, j].reshape([3, 1]))

            # Adding J^TJ to A:
            A += J.T.dot(J)

            # Converting to H, finding error E, and adding J^TE to b:
            H_pose_Wpt = np.linalg.inv(H_pose).dot(np.vstack((Wpts[:, j].reshape([3, 1]), 1)))
            HK = K.dot(H_pose_Wpt[:-1])
            f = HK / HK[-1]
            E = Ipts[:, j].reshape([2, 1]) - f[:-1]
            b += J.T.dot(E)
        E_pose += np.linalg.inv(A).dot(b)
    #------------------
    
    return hpose_from_epose(E_pose)

#----- Functions Go Below -----

def epose_from_hpose(T):
    """Euler pose vector from homogeneous pose matrix."""
    E = np.zeros((6, 1))
    E[0:3] = np.reshape(T[0:3, 3], (3, 1))
    E[3:6] = rpy_from_dcm(T[0:3, 0:3])
  
    return E

def hpose_from_epose(E):
    """Homogeneous pose matrix from Euler pose vector."""
    T = np.zeros((4, 4))
    T[0:3, 0:3] = dcm_from_rpy(E[3:6])
    T[0:3, 3] = np.reshape(E[0:3], (3,))
    T[3, 3] = 1
  
    return T
