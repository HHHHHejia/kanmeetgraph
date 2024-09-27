datasets=("bace" "bbbp" "clintox" "hiv" "sider" "tox21" "toxcast" "muv")
# 遍历每个 dataset
for dataset in "${datasets[@]}"
do
    python finetune.py \
        --runseed 0 \
        --dataset $dataset \
        --device 7 \
        --num_layer 2 \
        --emb_dim 64 \
        --kan_mlp kan\
        --kan_type eff\
        --input_model_file ./to_test/l2d64e100_eff/rgcl_seed0_100_gnn2.pth
done
