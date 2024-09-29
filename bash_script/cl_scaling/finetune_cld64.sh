#!/bin/bash

seed=$1
device=$2
kan_mlp=$3
model=$4
datasets=("bace" "bbbp" "clintox" "hiv" "sider" "tox21" "toxcast" "muv")
ratios="0.01 0.05 0.1 0.2 0.3 0.4 0.6 0.8 1"

# 遍历每个 dataset 和 ratio
for train_ratio in $ratios
do
    for dataset in "${datasets[@]}"
    do
        python finetune_ratio.py \
            --runseed $seed \
            --dataset $dataset \
            --device $device \
            --num_layer 2 \
            --emb_dim 64 \
            --kan_mlp $kan_mlp \
            ${model:+--input_model_file $model} \
            --data_fraction $train_ratio
    done
done
