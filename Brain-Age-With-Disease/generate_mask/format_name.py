import os,shutil

root_path = "generate_mask/test_img/True_mask"
target_path = "generate_mask/test_img/True_mask_format"

file_list = sorted(os.listdir(root_path))
print(file_list, len(file_list))

for idx in range(0, len(file_list)): 
    code_4 = '%04d'% idx  
    print(code_4, file_list[idx])
    shutil.copy(os.path.join(root_path,file_list[idx]), os.path.join(target_path,code_4 + ".nii.gz"))