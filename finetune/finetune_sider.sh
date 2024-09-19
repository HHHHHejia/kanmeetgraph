for runseed in 0 1 2 3 4 5 6 7 8 9
do
    for type in gcn gin gat graphsage
    do
        python finetune.py --runseed $runseed --dataset sider --device 5 --gnn_type $type
    done
done
        #path="./models_rgcl/no${no}/rgcl_seed0_100_gnn.pth"
        #python finetune.py --input_model_file $path --runseed $runseed --dataset sider --device 5