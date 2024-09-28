seed=$1
device=$2
datasets=("sider"  "tox21" "toxcast" "muv" "bace" "bbbp" "clintox" "hiv")


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
        --kan_type bsrbf\
        --input_model_file ./to_test/aug/l2d64kanbf/bf/rgcl_seed0_40_gnn2.pth
done
