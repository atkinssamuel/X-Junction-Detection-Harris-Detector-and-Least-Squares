import numpy as np
from saddle_point import saddle_point

# Build non-smooth but noise-free test patch.
Il = np.hstack((np.ones((10, 10)), np.zeros((10, 10)))) 
Ir = np.hstack((np.zeros((10, 10)), np.ones((10, 10))))
I = np.vstack((Il, Ir))

pt = saddle_point(I)
print('Saddle point is at: (%.2f, %.2f)' % (pt[0], pt[1]))