import numpy as np
from imageio import imread
from mat4py import loadmat
import matplotlib.pyplot as plt
from cross_junctions import cross_junctions
from matplotlib import patches
from matplotlib.path import Path
from PIL import Image

def plotResult(Ipts, I, bpoly):
    Ipts = Ipts.T
    bpoly = bpoly.T
    bpoly_list = (bpoly[0][0], bpoly[0][1]), (bpoly[1][0], bpoly[1][1]), (bpoly[2][0], bpoly[2][1]), \
                 (bpoly[3][0], bpoly[3][1]), (bpoly[0][0], bpoly[0][1])
    codes = [
        Path.MOVETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.CLOSEPOLY
    ]

    bounding_poly = Path(bpoly_list, codes)
    fig, ax = plt.subplots()
    patch = patches.PathPatch(bounding_poly, facecolor='none', lw=2)
    ax.add_patch(patch)
    for index in range(len(Ipts)):
        plt.scatter(Ipts[index][0], Ipts[index][1])
    plt.imshow(I)
    plt.show()

# Load the boundary.
# bpoly.shape = (2, 4)
bpoly = np.array(loadmat("bounds.mat")["bpolyh1"])
# Load the world points.
# Wpts.shape = (3, 48) - each column is a world point
Wpts = np.array(loadmat("world_pts.mat")["world_pts"])

# Load the example target image.
I = imread("example_target.png")
Ipts = cross_junctions(I, bpoly, Wpts)

# You can plot the points to check!
plotResult(Ipts, I, bpoly)