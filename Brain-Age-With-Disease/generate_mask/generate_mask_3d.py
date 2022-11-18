import numpy as np 
import random
from PIL import Image 
import cv2
import nibabel as nib 
import scipy.ndimage as sn 
from random import randint, seed 
from PIL import Image

# Code From https://stackoverflow.com/questions/46626267/how-to-generate-a-sphere-in-3d-numpy-array

def sphere(shape, radius, position):
    """Generate an n-dimensional spherical mask."""
    # assume shape and position have the same length and contain ints
    # the units are pixels / voxels (px for short)
    # radius is a int or float in px
    assert len(position) == len(shape)
    n = len(shape)
    semisizes = (radius,) * len(shape)

    # genereate the grid for the support points
    # centered at the position indicated by position
    grid = [slice(-x0, dim - x0) for x0, dim in zip(position, shape)]
    position = np.ogrid[grid]
    # calculate the distance of all points from `position` center
    # scaled by the radius
    arr = np.zeros(shape, dtype=float)
    for x_i, semisize in zip(position, semisizes):
        # this can be generalized for exponent != 2
        # in which case `(x_i / semisize)`
        # would become `np.abs(x_i / semisize)`
        arr += (x_i / semisize) ** 2

    # the inner part of the sphere will have distance below or equal to 1
    return arr <= 1.0

# this will save a sphere in a boolean array
# the shape of the containing array is: (256, 256, 256)
# the position of the center is: (127, 127, 127)
# if you want is 0 and 1 just use .astype(int)
# for plotting it is likely that you want that
arr = sphere((91, 109, 91), 3, (45, 50, 45)).astype(int)

# just for fun you can check that the volume is matching what expected
# (the two numbers do not match exactly because of the discretization error)

print(np.sum(arr))

arr = sphere((91, 109, 91), 3, (45, 50, 45)).astype(int)
print(arr.shape)
print(arr)

# plot in 3D
import matplotlib.pyplot as plt
from skimage import measure

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')

verts, faces, normals, values = measure.marching_cubes(arr, 0.5)
ax.plot_trisurf(
    verts[:, 0], verts[:, 1], faces, verts[:, 2], cmap='Spectral',
    antialiased=False, linewidth=0.0)
plt.show()