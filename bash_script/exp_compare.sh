# 从命令行参数传入变量
device=$1   # 第一个参数是 GPU 号
runseed=$2  # 第二个参数是 runseed
num_layer=$3  # 第三个参数是 num_layer
dataset="bace"
type="gin"

# emb_dim 的不同值列表
emb_dims=(8 16 32 64)

# 遍历 emb_dim 并顺序执行 mlp 和 kan 两种任务
for emb_dim in "${emb_dims[@]}"
do
    # 运行任务1：mlp
    echo "Running MLP on GPU $device with num_layer $num_layer and emb_dim $emb_dim"
    CUDA_VISIBLE_DEVICES=$device python finetune.py \
        --runseed $runseed \
        --dataset $dataset \
        --device $device \
        --num_layer $num_layer \
        --emb_dim $emb_dim \
        --gnn_type $type \
        --kan_mlp mlp

    # 运行任务2：kan with sum neuron_fun
    echo "Running KAN on GPU $device with num_layer $num_layer and emb_dim $emb_dim"
    CUDA_VISIBLE_DEVICES=$device python finetune.py \
        --runseed $runseed \
        --dataset $dataset \
        --device $device \
        --num_layer $num_layer \
        --emb_dim $emb_dim \
        --gnn_type $type \
        --kan_mlp kan \
        --kan_mp kan \
        --neuron_fun sum
done
