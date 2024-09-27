datasets=("bace" "bbbp" "clintox" "hiv" "sider" "tox21" "toxcast" "muv")
# 遍历每个 dataset
for dataset in "${datasets[@]}"
do
    python finetune.py \
        --runseed 0 \
        --dataset $dataset \
        --device 4 \
        --num_layer 2 \
        --emb_dim 128 \
        --kan_mlp kan\
        --input_model_file ./to_test/l2d128e20_noneg/rgcl_seed0_20_gnn1.pth
done
