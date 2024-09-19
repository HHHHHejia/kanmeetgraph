for runseed in 0 1 2 3 4 5 6 7 8 9
do
    for type in gcn gin gat graphsage
    do
        python finetune.py --runseed $runseed --dataset clintox --device 2 --gnn_type $type
    done
done
