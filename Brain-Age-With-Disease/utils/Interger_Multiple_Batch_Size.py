import torch
import numpy as np
class Integer_Multiple_Batch_Size(torch.utils.data.Dataset):
    #train_dataset
    def __init__(self, folder_dataset, batch_size=8):
        self.folder_dataset = folder_dataset
        self.batch_size = batch_size

        source_dataset_len = len(self.folder_dataset)
        num_need_to_complement = self.batch_size - (source_dataset_len % self.batch_size)
        
        idx_list = np.arange(0, source_dataset_len)#总共的长度
        complement_idx = idx_list[-num_need_to_complement:]#按不足的长度，从最后边进行了选择
        self.complemented_idx = np.concatenate([idx_list, complement_idx], axis=0)
        self.complemented_size = self.complemented_idx.shape[0]
        # print(self.complemented_idx.shape, self.complemented_size)
        
    def __len__(self):
        return self.complemented_size

    def __getitem__(self, index):
        return self.folder_dataset[self.complemented_idx[index]]