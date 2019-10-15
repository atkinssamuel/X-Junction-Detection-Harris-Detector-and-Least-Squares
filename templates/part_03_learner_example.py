import numpy as np
from find_jacobian import find_jacobian

# Set up test case - fixed parameters.
K = np.array([[564.9, 0, 337.3], [0, 564.3, 226.5], [0, 0, 1]])
Wpt = np.array([[0.0635, 0, 0]]).T

# Camera pose (rotation matrix, translation vector).
C_cam = np.array([[ 0.960656116714365, -0.249483426036932,  0.122056730876061],
                  [-0.251971275568189, -0.967721063070012,  0.005140075795822],
                  [ 0.116834505638601, -0.035692635424156, -0.992509815603182]])
t_cam = np.array([[0.201090356081375, 0.114474051344464, 1.193821106321156]]).T

Twc = np.hstack((C_cam, t_cam))
Twc = np.vstack((Twc, np.array([[0, 0, 0, 1]])))
J = find_jacobian(K, Twc, Wpt)
print(J)

# J =
# -477.1016  121.4005   43.3460  -18.8900  592.2179  -71.3193
#  130.0713  468.1394  -59.8803  578.8882  -14.6399  -49.5217