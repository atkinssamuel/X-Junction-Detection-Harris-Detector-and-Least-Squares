import numpy as np
from mat4py import loadmat
from pose_estimate_nls import pose_estimate_nls

# Camera intrinsics matrix - known.
K = np.array([[564.9, 0, 337.3], [0, 564.3, 226.5], [0, 0, 1]])

# Load landmark points (3D - ground truth).
Wpts = np.array(loadmat("world_pts.mat")["world_pts"])

# Load initial guess for camera pose.
camera = loadmat("camera_pose_01.mat")["camera"]["guess"]

# Load detected saddle points (2D - in image).
Ipts = np.array(loadmat("saddle_points.mat")["Ipts"])

# Homogeneous pose matrix (4 x 4).
Twcg = np.hstack((camera["C"], camera["t"]))
Twcg = np.vstack((Twcg, np.array([[0, 0, 0, 1]])))

Twc = pose_estimate_nls(K, Twcg, Ipts, Wpts)
print(Twc)

# Twc =
#     0.9159   -0.3804    0.1282    0.0932
#     0.3827    0.9239    0.0074   -0.0082
#    -0.1212    0.0423    0.9917   -1.0947
#          0         0         0    1.0000