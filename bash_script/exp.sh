device=0
runseed=0
type="gin"
datasets="tox21"
#kan_type=("ori", "bsrbf", "cheby", "eff", "fast", "faster", "fourier", "jacobi", "laplace", "legendre", "wavelet")



#euclid
python finetune.py --runseed 0 --dataset tox21 --device 0 --gnn_type gin --kan_mlp mlp
python finetune.py --runseed 0 --dataset tox21 --device 2 --gnn_type gin --kan_mlp kan --kan_type ori
python finetune.py --runseed 0 --dataset tox21 --device 6 --gnn_type gin --kan_mlp kan --kan_type bsrbf
python finetune.py --runseed 0 --dataset tox21 --device 7 --gnn_type gin --kan_mlp kan --kan_type cheby

#gauss
python finetune.py --runseed 0 --dataset tox21 --device 0 --gnn_type gin --kan_mlp kan --kan_type eff --grid 10 --k 5
python finetune.py --runseed 0 --dataset tox21 --device 1 --gnn_type gin --kan_mlp kan --kan_type fast --grid 10
python finetune.py --runseed 0 --dataset tox21 --device 2 --gnn_type gin --kan_mlp kan --kan_type faster --grid 10
python finetune.py --runseed 0 --dataset tox21 --device 3 --gnn_type gin --kan_mlp kan --kan_type fourier --grid 10

#boole
python finetune.py --runseed 0 --dataset tox21 --device 0 --gnn_type gin --kan_mlp kan --kan_type jacobi
python finetune.py --runseed 0 --dataset tox21 --device 1 --gnn_type gin --kan_mlp kan --kan_type laplace
python finetune.py --runseed 0 --dataset tox21 --device 2 --gnn_type gin --kan_mlp kan --kan_type legendre
python finetune.py --runseed 0 --dataset tox21 --device 3 --gnn_type gin --kan_mlp kan --kan_type wavelet





