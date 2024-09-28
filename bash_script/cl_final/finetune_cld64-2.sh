seed=$1
device=$2
datasets=("sider" "tox21" "toxcast" "muv" "bace" "bbbp" "clintox" "hiv"  )
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
        --kan_type eff\
        --input_model_file ./to_test/aug/l2d64kaneff/eff/rgcl_seed0_20_gnn2.pth
done
