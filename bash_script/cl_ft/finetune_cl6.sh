datasets=("bace" "bbbp" "clintox" "hiv" "sider" "tox21" "toxcast" "muv")
# 遍历每个 dataset
for dataset in "${datasets[@]}"
do
    python finetune.py \
        --runseed 0 \
        --dataset $dataset \
        --device 6 \
        --num_layer 2 \
        --emb_dim 256 \
        --kan_mlp kan\
        --input_model_file ./to_test/l2d256e20/rgcl_seed0_20_gnn1.pth
done
