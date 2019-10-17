import numpy as np
import matplotlib.pyplot as plt
from saddle_point import saddle_point
from imageio import imread
from scipy.ndimage.filters import *
from PIL import Image
from cross_junctions import harris_corner_detector

# Build non-smooth but noise-free test patch.
Il = np.hstack((np.ones((10, 10)), np.zeros((10, 10)))) 
Ir = np.hstack((np.zeros((10, 10)), np.ones((10, 10))))
I = np.vstack((Il, Ir))

I = Image.open('image_splice29.533998529297772.png').convert('L')
I = np.array(I)
K = np.array([[1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1]])/10


I = convolve(I, K)

pts = harris_corner_detector(I, None)
pt = saddle_point(I)
plt.imshow(I, cmap='gray')
plt.scatter(pt[0], pt[1], c='g')
plt.scatter(pts[:, 0], pts[:, 1], c='r')
plt.show()
print('Saddle point is at: (%.2f, %.2f)' % (pt[0], pt[1]))