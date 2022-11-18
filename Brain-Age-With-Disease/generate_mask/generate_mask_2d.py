#import argparse
import numpy as np 
import random
from PIL import Image 
import cv2
import nibabel as nib 
import scipy.ndimage as sn 
from random import randint, seed 
from PIL import Image


def _generate_mask(a, b, c, d):
    """_summary_

    Args:
        a (_type_): _description_
        b (_type_): _description_
        c (_type_): _description_
        d (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    # Generate a zeros array with Size 128x128x3 
    img=np.zeros((128,128,3),np.uint8) 

    # Generate a random x,y, which random range is from (a,b), (c,d)
    x1, y1 = randint(a, b), randint(c, d) 
    
    # 
    s1, s2 = randint(10, 20), randint(15,20)
    #a1, a2, a3 = randint(3, 180), randint(3, 180), randint(3, 180) 
    a1=randint(3,180)
    thickness=-1 #randint(90,95)
    
    # Generate a ellipse to img with parameter (x1,y1), (s1,s2), a1, thickness
    cv2.ellipse(img, (x1,y1), (s1,s2), a1, 0, 360,(1,1,1), thickness) 
    img=img
    
    # return a img,which pixels in ellipse is 1, pixels out of ellipse is 0
    return 1-img 

if __name__=="__main__":
    #l=list()
    root_img_path =  "/data/ziyang/workspace/brain_age_prediction/data/NC/combine/18/test"
    for i in range(1,100):
        a=_generate_mask(30, 95, 30, 95) 
        print(a)
        #l.append(a)
        #mask=np.array(l)
        img = Image.fromarray(a * 255).convert('1')
        # img.save('mask_img_test_2d/mask_%s.png'%i) 

    for i in range(100,166):
        a=_generate_mask(59, 69, 64, 69) 
        #l.append(a)
        #mask=np.array(l)
        img = Image.fromarray(a * 255).convert('1')
        # img.save('mask_img_test_2d/mask_%s.png'%i) 
        
        
