device=3
runseed=3
type="gin"

for kan_type in ori 
#for dataset in bace bbbp clintox sider tox21 toxcast hiv muv

#todo; hiv, 
#todo: muv,全部
for dataset in muv 
    #mlp baseline
    do
        #python finetune.py --runseed $runseed --dataset $dataset --device $device --gnn_type $type --kan_mlp mlp
        python finetune.py --runseed $runseed --dataset $dataset --device $device --gnn_type $type --kan_mlp kan --neuron_fun sum 
        python finetune.py --runseed $runseed --dataset $dataset --device $device --gnn_type $type --kan_mlp kan --neuron_fun mean 
    done


