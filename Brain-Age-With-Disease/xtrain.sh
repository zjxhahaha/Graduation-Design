#!/bin/bash

model=res18
batch_size=16
loss=mse
aux_loss=l1
lambd=0.0
GAMMA=2.0
MASK=cube_mask_larger   #normal   masked_img  
sorter=./network/Sodeep_pretrain_weight/Tied_rank_best_lstmla_slen_${batch_size}.pth.tar

save_path=./ckpt/T1_mask/${model}_${MASK}_crop_img_loss_${loss}_LAMBDA_${lambd}__aux_loss_${aux_loss}_GAMMA_${GAMMA}/

train_folder=Data/T1_mask/Train/
valid_folder=Data/T1_mask/Valid/
test_folder=Data/T1_mask/Test/
excel_path=Data/combine.xls


CUDA_VISIBLE_DEVICES=0     python main.py       \
--batch_size               $batch_size           \
--epochs                   100                   \
--lr                       1e-3                  \
--loss                     ${loss}               \
--aux_loss                 ${aux_loss}           \
--lambd                    $lambd                \
--gamma                    ${GAMMA}              \
--model                    ${model}              \
--output_dir               ${save_path}          \
--sorter                   ${sorter}             \
--schedular                step                  \
--train_folder             ${train_folder}       \
--valid_folder             ${valid_folder}       \
--test_folder              ${test_folder}        \
--excel_path               ${excel_path}         \
--accumulation_steps       1                     \
--mask_type                ${MASK}               \
