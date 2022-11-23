import os
import torch, random
import nibabel as nib
import numpy as np
from random import randint, seed


def cube_mask(image, number_cube_range=[1, 5], cube_size=[5, 30]):
    iterations = random.randint(number_cube_range[0], number_cube_range[1])
    masked_img = image
    mask = np.zeros(image.shape)

    for i in range(iterations):
        random_center_x, random_center_y, random_center_z = randint(20, 70), randint(20, 100), randint(20, 70)
        random_width, random_height, random_deep = randint(cube_size[0], cube_size[1]), randint(cube_size[0],
                                                                                                cube_size[1]), randint(
            cube_size[0], cube_size[1])
        mask = mask + cube(image.shape, (random_width, random_height, random_deep),
                           (random_center_x, random_center_y, random_center_z))
        # print((random_center_x, random_center_y, random_center_z),np.sum(mask))
        # masked_img = masked_img * mask
    mask = 1 - mask
    masked_img = masked_img * mask

    return masked_img


def cube(img_shape, mask_shape, position):
    mask = np.zeros(img_shape)
    # print('mask shape', mask.shape)
    width, height, deep = mask_shape[0], mask_shape[1], mask_shape[2]
    center_x, center_y, center_z = position[0], position[1], position[2]
    mask[center_x - width // 2: center_x + width // 2
    , center_y - height // 2: center_y + height // 2
    , center_z - deep // 2: center_z + deep // 2] = 1
    return mask