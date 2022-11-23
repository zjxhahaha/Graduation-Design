from tkinter import Y
import numpy as np 
import random,os
from PIL import Image 
# import cv2
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
    # print(semisizes)
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

# mask可以理解为一个矩阵
def cube(img_shape, mask_shape, position):
    #图像的形状，mask的形状，mask的位置
    mask = np.zeros(img_shape )
    # print('mask shape', mask.shape)
    width, height, deep = mask_shape[0], mask_shape[1], mask_shape[2]
    center_x, center_y, center_z = position[0], position[1], position[2]
    mask[ center_x - width // 2 : center_x + width // 2
        , center_y - height // 2: center_y + height // 2
        , center_z - deep // 2 : center_z + deep // 2] = 1
    return mask 
    
    

if __name__=="__main__":
    #l=list()
    root_img_path = "/opt/zhaojinxin/TSAN/brain_age_estimation_transfer_learning/train/"
    save_path = "/opt/zhaojinxin/TSAN/TSAN_Cube_mask_data/"
    file_list = os.listdir(root_img_path)
    N = 0
    #用于循环获得相当应的每一个图像
    for idx in range(0, len(file_list)):
    # for idx in range(0, 1):
        img = nib.load(os.path.join(root_img_path, file_list[idx]))
        affine = img.affine
        #对图形进行仿射  变换
        img_data = img.get_fdata()
        img_shape = img_data.shape
        # print("image shape:",img_shape)

        iterations = random.randint(1, 5)
        # print('number of mask:', iterations)

        masked_img = img_data
        #此处的iteration是有几个mask
        for i in range(iterations):
            #半径，中心坐标均是随机生成
            random_radius = randint(5,20)
            random_center_x, random_center_y, random_center_z = randint(20, 70), randint(20, 100), randint(20, 70) 
            # mask = sphere(img_shape, random_radius, (random_center_x, random_center_y, random_center_z))
            mask = cube(img_shape,(random_radius, random_radius,random_radius), (random_center_x, random_center_y, random_center_z))
            mask = 1 - mask
            #mask只有特定位置会实现赋值，因此1-，就会把要遮住的地方设置为0
            # print((random_center_x, random_center_y, random_center_z),random_radius, np.sum(mask))
            masked_img = masked_img * mask
            #这里就是得到最后的矩阵，因为实现了一个乘法
        name = file_list[idx].replace('.nii.gz', '_mask.nii.gz')
        nib.Nifti1Image(masked_img,affine).to_filename(os.path.join(save_path, name))
        N += 1
        print(N)
         

