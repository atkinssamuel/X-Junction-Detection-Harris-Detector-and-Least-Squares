import numpy as np
from imageio import imread
from mat4py import loadmat
from cross_junctions import cross_junctions

# Load the boundary.
bpoly = np.array(loadmat("bounds.mat")["bpolyh1"])

# Load the world points.
Wpts = np.array(loadmat("world_pts.mat")["world_pts"])

# Load the example target image.
I = imread("example_target.png")

Ipts = cross_junctions(I, bpoly, Wpts)

# You can plot the points to check!
print(Ipts)