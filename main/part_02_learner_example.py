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
    plt.savefig('result.png')
    plt.show()

basic_test = False

if basic_test == True:
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

extra_tests = True

if extra_tests == True:
    # Extra tests:
    I_files = []
    target_strings = ['0350', '0458', '0507', '0617', '0634', '0794', '0818', '0999', '1017', '1116', '1284', '1354',
                      '1391', '1500', '1577', '1615', '1671']
    examining_strings = target_strings
    bboxes = loadmat("bboxes.mat")
    Wpts = np.array(loadmat("../targets/world_pts.mat")["world_pts"])
    iteration = 0
    for target_string in target_strings:
        iteration += 1
        if target_string not in examining_strings:
            continue
        image = imread('../targets/image_00' + target_string + '.png')
        bbox = np.array(bboxes['image_00' + target_string])

        Ipts, inner_border = cross_junctions(image, bbox, Wpts)
        Ipts = Ipts.T
        fig, ax = plt.subplots()
        patch = patches.PathPatch(inner_border, facecolor='none', lw=2)
        ax.add_patch(patch)
        for index in range(len(Ipts)):
            plt.scatter(Ipts[index][0], Ipts[index][1])

        plt.imshow(image, cmap="gray")
        plt.title('image_00' + target_string)
        plt.show()
        plt.savefig('../results/result_00{}.png'.format(target_string))



