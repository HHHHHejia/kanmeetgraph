#!/bin/bash

# 从命令行参数传入变量
device=$1   # 第一个参数是 GPU 号
runseed=$2  # 第二个参数是 runseed
type="gin"

# 定义数据集列表
datasets=("bace" "bbbp" "clintox" "hiv" "sider" "tox21" "toxcast" "muv")

# num_layer 和 emb_dim 的不同值列表
num_layers=(2 3 4 5)
emb_dims=(8 16 32 64)

# 遍历每个 dataset
for dataset in "${datasets[@]}"
do
            # 运行任务1：mlp
            echo "Running MLP on GPU $device with dataset $dataset, num_layer $num_layer, emb_dim $emb_dim, and runseed $runseed"
            python finetune.py \
                --runseed $runseed \
                --dataset $dataset \
                --device $device \
                --num_layer $num_layer \
                --emb_dim $emb_dim \
                --gnn_type $type \
                --kan_mlp mlp
done
