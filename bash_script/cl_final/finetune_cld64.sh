seed=$1
device=$2
datasets=("bace" "bbbp" "clintox" "hiv" "sider" "tox21" "toxcast" "muv")
# 遍历每个 dataset
for dataset in "${datasets[@]}"
do
    python finetune.py \
        --runseed $seed \
        --dataset $dataset \
        --device $device \
        --num_layer 2 \
        --emb_dim 64 \
        --kan_mlp kan\
        --input_model_file ./to_test/aug/l2d64kanbf/kan/rgcl_seed0_40_gnn1.pth
done
