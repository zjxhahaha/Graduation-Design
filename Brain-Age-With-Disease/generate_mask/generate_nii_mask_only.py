from tkinter import Y
import numpy as np 
import random,os
from PIL import Image 
import cv2
import nibabel as nib 
import scipy.ndimage as sn 
from random import randint, seed 
from PIL import Image
import nibabel as nib


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


def cube(img_shape, mask_shape, position):
    mask = np.zeros(img_shape )
    print('mask shape', mask.shape)
    width, height, deep = mask_shape[0], mask_shape[1], mask_shape[2]
    center_x, center_y, center_z = position[0], position[1], position[2]
    mask[ center_x - width // 2 : center_x + width // 2
        , center_y - height // 2: center_y + height // 2
        , center_z - deep // 2 : center_z + deep // 2] = 1
    return mask 
    
    

if __name__=="__main__":
    #l=list()
    # root_img_path = "Data/T1_mask/Test/T1/"
    # save_path = "Data/T1_mask/Test/cube_mask_larger/"
    root_img_path = "/opt/zhaojinxin/TSAN/brain_age_estimation_transfer_learning/train/"
    save_path = "/opt/zhaojinxin/TSAN/TSAN_Cube_mask_data/"
    file_list = os.listdir(root_img_path)
    N = 0
    for idx in range(0, len(file_list)):
        img = nib.load(os.path.join(root_img_path, file_list[idx]))
        affine = img.affine
        img_data = img.get_fdata()
        img_shape = img_data.shape
        print("image shape:",img_shape)
        
        iterations = random.randint(1, 8)
        print('number of mask:', iterations)

        masked_img = img_data
        mask = np.zeros(img_shape)#生成一个大小相同的全零矩阵

        for i in range(iterations):
            random_radius = randint(5,20)
            random_center_x, random_center_y, random_center_z = randint(20, 70), randint(20, 100), randint(20, 70) 
            # mask = sphere(img_shape, random_radius, (random_center_x, random_center_y, random_center_z))
            #生成矩形
            random_width, random_height, random_deep = randint(5,30), randint(5,30), randint(5,30)
            #这里mask可能有点文件，也就是说会产生大于1的数字
            mask =  mask + cube(img_shape,(random_width, random_height, random_deep), (random_center_x, random_center_y, random_center_z))
            print((random_center_x, random_center_y, random_center_z),np.sum(mask))
            # masked_img = masked_img * mask 
        mask = 1 - mask
        name = file_list[idx].replace('.nii.gz', '_masked.nii.gz')
        nib.Nifti1Image(mask,affine).to_filename(os.path.join(save_path, name))
        N += 1
        print(N)
         

