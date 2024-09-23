#!/bin/bash
runseed=0
# 从命令行参数传入变量
device=$1   # 第一个参数是 GPU 号
dataset=$2
type="gin"


# num_layer 和 emb_dim 的不同值列表
num_layers=(2 3 4 5)
emb_dims=(8 16 32 64)


# 遍历 num_layer 和 emb_dim 并顺序执行 mlp 和 kan 两种任务
for num_layer in "${num_layers[@]}"
do
    for emb_dim in "${emb_dims[@]}"
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
            --kan_mlp mlp\
            --use_transformer mlp

        # 运行任务2：kan with sum neuron_fun
        echo "Running KAN on GPU $device with dataset $dataset, num_layer $num_layer, emb_dim $emb_dim, and runseed $runseed"
        python finetune.py \
            --runseed $runseed \
            --dataset $dataset \
            --device $device \
            --num_layer $num_layer \
            --emb_dim $emb_dim \
            --gnn_type $type \
            --kan_mlp kan \
            --use_transformer kan
    done
done
