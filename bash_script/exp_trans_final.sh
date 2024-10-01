#!/bin/bash
type="gin"
device=$1   # 第一个参数是 GPU 号
num_layer=$2
emb_dim=$3

# num_layer 和 emb_dim 的不同值列表
datasets=("bace" "bbbp" "clintox" "hiv" "sider" "tox21" "toxcast" "muv")
for dataset in "${datasets[@]}"
do
    for runseed in 0 1 2
    do
        echo "Running MLP on GPU $device with dataset $dataset, num_layer $num_layer, emb_dim $emb_dim, and runseed $runseed"
        python finetune.py \
            --runseed $runseed \
            --dataset $dataset \
            --device $device \
            --num_layer $num_layer \
            --emb_dim $emb_dim \
            --gnn_type $type \
            --kan_mlp mlp\
            --use_transformer kan\
            --grid 5\
            --k 3 > "./log/final_log1.log" 2>&1 &
    done
    wait
done