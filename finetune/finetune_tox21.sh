for runseed in 0 1 2 3 4 5 6 7 8 9
do
    for type in gcn gin gat graphsage
    do
        python finetune.py --runseed $runseed --dataset tox21 --device 6 --gnn_type $type
    done
done
