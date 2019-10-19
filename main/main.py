import numpy as np
from imageio import imread
from mat4py import loadmat
import matplotlib.pyplot as plt
from matplotlib import patches
from cross_junctions import cross_junctions

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
    #plt.show()
    plt.savefig('../results/result_00{}.png'.format(target_string))



