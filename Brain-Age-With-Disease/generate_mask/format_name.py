import os,shutil

root_path = "generate_mask/test_img/True_mask"
root_path = "/opt/zhaojinxin/TSAN/brain_age_estimation_transfer_learning/train/"
target_path = "generate_mask/test_img/True_mask_format"

#升序？
file_list = sorted(os.listdir(root_path))
print("file_list", len(file_list))

for idx in range(0, len(file_list)):
# for idx in range(0, 1):
    code_4 = '%04d'% idx
#此处含义是code_4表示4位的序号
    print(code_4, file_list[idx])
#含义是将名字给改变，变成0001~1234这种序号
    shutil.copy(os.path.join(root_path,file_list[idx]), os.path.join(target_path,code_4 + ".nii.gz"))