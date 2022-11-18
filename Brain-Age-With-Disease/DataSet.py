import os
import torch
import nibabel as nib
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import torchio as tio 
from utils.Crop_and_padding import resize_image_with_crop_or_pad
from utils.Interger_Multiple_Batch_Size import Integer_Multiple_Batch_Size
from utils.cube_mask import cube_mask


def nii_loader(path):
    img = nib.load(str(path))
    data = img.get_fdata()
    return data

def read_table(path):
    return(pd.read_excel(path).values) # default to first sheet

def white0(image, threshold=0):
    "standardize voxels with value > threshold"
    image = image.astype(np.float32)
    mask = (image > threshold).astype(int)

    image_h = image * mask
    image_l = image * (1 - mask)

    mean = np.sum(image_h) / np.sum(mask)
    std = np.sqrt(np.sum(np.abs(image_h - mean)**2) / np.sum(mask))

    if std > 0:
        ret = (image_h - mean) / std + image_l
    else:
        ret = image * 0.
    return ret

class IMG_Folder(torch.utils.data.Dataset):
    def __init__(self,excel_path, data_path, loader=nii_loader,transforms=None):
        self.root = data_path
        self.sub_fns = sorted(os.listdir(self.root))
        self.table_refer = read_table(excel_path)
        self.loader = loader
        self.transform = transforms

    def __len__(self):
        return len(self.sub_fns)

    def __getitem__(self,index):
        sub_fn = self.sub_fns[index]
        for f in self.table_refer:
            
            sid = str(f[0])[:12]
            slabel = (int(f[1]))
            smale = f[2]
            if sid not in sub_fn:
                continue
            sub_path = os.path.join(self.root, sub_fn)
            img = self.loader(sub_path)
            img = white0(img)
            
            img = np.expand_dims(img, axis=0)
            img = np.ascontiguousarray(img, dtype= np.float32)
            img = torch.from_numpy(img).type(torch.FloatTensor)

            if self.transform is not None:
                img = self.transform(img)
            break
        slabel = torch.tensor(np.array(slabel))
        slabel = torch.unsqueeze(slabel, dim=0)
        return (img, sid, slabel, smale)

class IMG_Folder_with_mask(torch.utils.data.Dataset):
    def __init__(self,excel_path, data_path, loader=nii_loader,transforms=None, mask_type='cube_mask'):
        self.root = data_path
        self.mask_type = mask_type
        self.sub_fns_img  = sorted(os.listdir(os.path.join(self.root, 'T1')))
        self.sub_fns_mask = sorted(os.listdir(os.path.join(self.root, self.mask_type)))

        self.table_refer = read_table(excel_path)
        self.loader = loader
        self.transform = transforms

    def __len__(self):
        return len(self.sub_fns_img)

    def __getitem__(self,index):
        sub_fn_img = self.sub_fns_img[index]
        sub_fn_mask = self.sub_fns_mask[index]
        
        assert sub_fn_img[:12] == sub_fn_mask[:12]

        for f in self.table_refer:
            
            sid = str(f[0])[:12]
            slabel = (int(f[1]))
            smale = f[2]
            if sid not in sub_fn_img:
                continue
            sub_img_path = os.path.join(self.root, 'T1', sub_fn_img)
            # sub_mask_path = os.path.join(self.root, self.mask_type, sub_fn_mask)
            img = self.loader(sub_img_path)
            img = white0(img)
            
            # mask = self.loader(sub_mask_path)
            # masked_img = img * mask
            # img = resize_image_with_crop_or_pad(img, img_size=(72,96,72),  mode='symmetric')
            # masked_img = resize_image_with_crop_or_pad(masked_img, img_size=(72,96,72), mode='symmetric')
            
            img = np.expand_dims(img, axis=0)
            img = np.ascontiguousarray(img, dtype= np.float32)
            img = torch.from_numpy(img).type(torch.FloatTensor)
            
            masked_img = tio.Lambda(cube_mask)(img).type(torch.FloatTensor)
            # masked_img = np.expand_dims(masked_img, axis=0)
            # masked_img = np.ascontiguousarray(masked_img, dtype= np.float32)
            # masked_img = torch.from_numpy(masked_img).type(torch.FloatTensor)

            if self.transform is not None:
                img = self.transform(img)
                masked_img = self.transform(masked_img)
            break
        slabel = torch.tensor(np.array(slabel))
        slabel = torch.unsqueeze(slabel, dim=0)
        return (img, masked_img, sid, slabel, smale)


if __name__ == "__main__":
    excel_path ="/data/ziyang/workspace/Age-Estimation/brain_age_prediction/lables/combine.xls"
    train_folder = "/data/ziyang/workspace/Age-Estimation/brain_age_prediction/data/NC/combine/T1_mask/Train/"
    
    transforms = tio.Compose(
        [
        #  tio.RandomAffine(scales=0.75),
        #  tio.RandomFlip(flip_probability=0.5, axes=('LR',)),
        #  tio.RandomBiasField(coefficients=1),
        #  tio.RandomSpike(),
        #  tio.RandomNoise(),
        #  tio.RandomBlur()
        ]
    )
    train_dataset = IMG_Folder_with_mask(excel_path,train_folder
                               , transforms=transforms
                               )
    train_dataset = Integer_Multiple_Batch_Size(train_dataset, batch_size=8)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True,
                               drop_last=False, num_workers=24)
    print(len(train_loader))
    for idx, (img,masked_img, sid, target, male) in enumerate(train_loader):
    
        # =========== convert male lable to one hot type =========== #
        
        target = target.type(torch.FloatTensor)
        print(img.shape, masked_img.shape, target.shape, target)