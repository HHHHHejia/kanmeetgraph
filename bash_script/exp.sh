device=0
runseed=0
type="gin"

#for dataset in bace bbbp clintox hiv sider tox21 toxcast 
for dataset in muv 
    #mlp baseline
    do
        #python finetune.py --runseed $runseed --dataset $dataset --device $device --gnn_type $type --kan_mlp mlp
        #python finetune.py --runseed $runseed --dataset $dataset --device $device --gnn_type $type --kan_mlp kan --neuron_fun mean 
        python finetune.py --runseed $runseed --dataset $dataset --device $device --gnn_type $type --kan_mlp kan --neuron_fun sum 
    done