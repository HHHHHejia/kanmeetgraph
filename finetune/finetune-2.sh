split=scaffold
for runseed in 0 1 2 3 4 5 6 7 8 9
do
for dataset in bace bbbp clintox hiv muv sider tox21 toxcast
do
python finetune.py --input_model_file ./models_rgcl/no37/rgcl_seed0_80_gnn.pth --split $split --runseed $runseed --dataset $dataset --device 2
done
done
