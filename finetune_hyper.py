import argparse

from loader import MoleculeDataset
from torch_geometric.loader import DataLoader
from torchinfo import summary

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np

from model import GNN_graphpred
from sklearn.metrics import roc_auc_score

from splitters import scaffold_split, random_split, random_scaffold_split
import pandas as pd

import os
import shutil
import wandb

criterion = nn.BCEWithLogitsLoss(reduction="none")
# torch.set_float32_matmul_precision('high')

args = None  # Global variable to hold command-line arguments

def train_epoch(model, device, loader, optimizer):
    model.train()

    for step, batch in enumerate(loader):
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        y = batch.y.view(pred.shape).to(torch.float64)

        # Whether y is non-null or not.
        is_valid = y**2 > 0
        # Loss matrix
        loss_mat = criterion(pred.double(), (y+1)/2)
        # Loss matrix after removing null target
        loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
            
        optimizer.zero_grad()
        loss = torch.sum(loss_mat)/torch.sum(is_valid)
        loss.backward()

        optimizer.step()


def eval_model(model, device, loader):
    model.eval()
    y_true = []
    y_scores = []

    for step, batch in enumerate(loader):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        y_true.append(batch.y.view(pred.shape))
        y_scores.append(pred)

    y_true = torch.cat(y_true, dim = 0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim = 0).cpu().numpy()

    roc_list = []
    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == -1) > 0:
            is_valid = y_true[:,i]**2 > 0
            roc_list.append(roc_auc_score((y_true[is_valid,i] + 1)/2, y_scores[is_valid,i]))

    if len(roc_list) < y_true.shape[1]:
        print("Some target is missing!")
        print("Missing ratio: %f" %(1 - float(len(roc_list))/y_true.shape[1]))

    return sum(roc_list)/len(roc_list) # y_true.shape[1]


def run_training():
    global args
    # Initialize wandb
    wandb.init(project="KAGNN")
    config = wandb.config

    # Override hyperparameters with wandb.config
    args.num_layer = config.num_layer
    args.emb_dim = config.emb_dim
    args.grid = config.grid
    args.k = config.k

    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)

    # Bunch of classification tasks
    if args.dataset == "tox21":
        num_tasks = 12
        epoch_num = 100
    elif args.dataset == "hiv":
        num_tasks = 1
        epoch_num = 100
    elif args.dataset == "pcba":
        num_tasks = 128
        epoch_num = 100
    elif args.dataset == "muv":
        num_tasks = 17
        epoch_num = 50
    elif args.dataset == "bace":
        num_tasks = 1
        epoch_num = 100
    elif args.dataset == "bbbp":
        num_tasks = 1
        epoch_num = 100
    elif args.dataset == "toxcast":
        num_tasks = 617
        epoch_num = 100
    elif args.dataset == "sider":
        num_tasks = 27
        epoch_num = 100
    elif args.dataset == "clintox":
        num_tasks = 2
        epoch_num = 300
    else:
        raise ValueError("Invalid dataset name.")

    dataset = MoleculeDataset("dataset/" + args.dataset, dataset=args.dataset)

    print(args.dataset)

    if args.split == "scaffold":
        smiles_list = pd.read_csv('dataset/' + args.dataset + '/smiles.csv', header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1)
        print("scaffold")
    elif args.split == "random":
        train_dataset, valid_dataset, test_dataset = random_split(dataset, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed = args.seed)
        print("random")
    elif args.split == "random_scaffold":
        smiles_list = pd.read_csv('dataset/' + args.dataset + '/smiles.csv', header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = random_scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed = args.seed)
        print("random scaffold")
    else:
        raise ValueError("Invalid split option.")

    print(train_dataset[0])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Set up model
    model = GNN_graphpred(args.num_layer, args.emb_dim, num_tasks,
                           JK=args.JK, drop_ratio=args.dropout_ratio, 
                           graph_pooling=args.graph_pooling, gnn_type=args.gnn_type,
                            kan_mlp = args.kan_mlp, grid = args.grid, k = args.k)
    if not args.input_model_file == "":
        print(args.input_model_file)
        model.from_pretrained(args.input_model_file, device)
    model.to(device)

    # Set up optimizer
    model_param_group = []
    model_param_group.append({"params": model.gnn.parameters()})

    if args.graph_pooling == "attention":
        model_param_group.append({"params": model.pool.parameters(), "lr": args.lr*args.lr_scale})

    model_param_group.append({"params": model.graph_pred_linear.parameters(), "lr": args.lr*args.lr_scale})
    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
    print(optimizer)
    summary(model)

    train_acc_list = []
    val_acc_list = []
    test_acc_list = []

    for epoch in range(1, epoch_num+1):
        print("====epoch " + str(epoch))
        
        train_epoch(model, device, train_loader, optimizer)

        print("====Evaluation")
        if args.eval_train:
            train_acc = eval_model(model, device, train_loader)
        else:
            print("omit the training accuracy computation")
            train_acc = 0
        val_acc = eval_model(model, device, val_loader)
        test_acc = eval_model(model, device, test_loader)

        val_acc_list.append(val_acc)
        test_acc_list.append(test_acc)
        train_acc_list.append(train_acc)

        print("train: %.4f val: %.4f test: %.4f" %
              (train_acc, val_acc, test_acc))

        print("")
        
        # Log metrics to wandb
        wandb.log({
            'train_acc': train_acc,
            'val_acc': val_acc,
            'test_acc': test_acc,
        })


def main():
    global args
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--lr_scale', type=float, default=1,
                        help='relative learning rate for the feature extraction layer (default: 1)')
    parser.add_argument('--decay', type=float, default=1e-4,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=30,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0.1,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--graph_pooling', type=str, default="mean",
                        help='graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--gnn_type', type=str, default="gin", help = "gin/gcn/gat/graphsage")
    parser.add_argument('--dataset', type=str, default='bbbp', help='root directory of dataset. For now, only classification.')
    parser.add_argument('--input_model_file', type=str, default="", help='filename to read the model (if there is any)')
    parser.add_argument('--filename', type=str, default='', help='output filename')
    parser.add_argument('--seed', type=int, default=42, help = "Seed for splitting the dataset.")
    parser.add_argument('--runseed', type=int, default=0, help = "Seed for minibatch selection, random initialization.")
    parser.add_argument('--split', type=str, default="scaffold", help = "random or scaffold or random_scaffold")
    parser.add_argument('--eval_train', type=int, default=1, help='evaluating training or not')
    parser.add_argument('--num_workers', type=int, default=32, help='number of workers for dataset loading')
    parser.add_argument('--kan_mlp', type = str, default='mlp', help="mlp or kan")
    parser.add_argument('--grid', type = int, default = 3, help="bspline grid")
    parser.add_argument('--k', type = int, default = 1, help="bspline order")

    args = parser.parse_args()

    # Define sweep configuration
    sweep_config = {
        'method': 'random',  # Random search
        'metric': {
            'name': 'test_acc',
            'goal': 'maximize'   
        },
        'parameters': {
            'num_layer': {
                'values': [2, 3, 4]
            },
            'emb_dim': {
                'values': [8, 16, 32, 64]
            },
            'dropout_ratio': {
                'values': [0.1,0.2, 0.3, 0.4, 0.5]
            },
            'grid': {
                'values': [1,2,3,4,5,6,7,8,9,10]
            },
            'k':{
                'values': [1,2,3,4,5]
            }
        }
    }

    # Initialize sweep
    #sweep_id = wandb.sweep(sweep_config, project="KAGNN")
    sweep_id = 'b4xmxn8g'
    # Start the sweep
    wandb.agent(sweep_id, function=run_training, project="KAGNN")

if __name__ == "__main__":
    main()
