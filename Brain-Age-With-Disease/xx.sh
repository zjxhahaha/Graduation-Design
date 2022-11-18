#!/bin/bash

bash xtrain.sh  agedb   resnet50    128     mse   matrix_l1   0.1
wait
bash xtrain.sh  agedb   resnet50    128     mse   matrix_l1   0.2
wait
bash xtrain.sh  agedb   resnet50    128     mse   matrix_l1   0.5
wait
bash xtrain.sh  agedb   resnet50    128     mse   matrix_l1   0.8
wait


# ===============================================================
# bash xtrain.sh  imdb_wiki   scaledense    128     mse   ranking   10.0
# wait

