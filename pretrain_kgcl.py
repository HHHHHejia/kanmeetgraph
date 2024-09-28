import argparse
import argparse
from loader import MoleculeDataset_aug
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary

import numpy as np
from tqdm import tqdm
import os

from model import GNN

import gc
import wandb
from datetime import datetime
import pytz
from copy import deepcopy



class graphcl(nn.Module):
    def __init__(self, gnn, args):
        super(graphcl, self).__init__()
        self.gnn = gnn
        self.pool = global_mean_pool
        self.emb_dim = args.emb_dim
        self.projection_head = nn.Sequential(nn.Linear(self.emb_dim, self.emb_dim), nn.ReLU(inplace=True), nn.Linear(self.emb_dim, self.emb_dim))

    def forward_cl(self, x, edge_index, edge_attr, batch):
        x = self.gnn(x, edge_index, edge_attr, batch)
        x = self.pool(x, batch)
        x = self.projection_head(x)
        return x

    def loss_cl(self, x1, x2, temp, args):
        T = temp
        batch_size, _ = x1.size()
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        if args.no_pos:
            loss = 1 / (sim_matrix.sum(dim=1) - pos_sim)
        elif args.no_neg:
            loss = pos_sim
        else:    
            loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()
        return loss

    def loss_infonce(self, x1, x2, temp):
        T = temp
        batch_size, _ = x1.size()
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / sim_matrix.sum(dim=1)
        loss = - torch.log(loss).mean()
        return loss

def train(args, model1, model2, device, dataset, optimizer1, optimizer2):
    dataset.aug = "none"

    dataset1 = dataset.shuffle()
    dataset2 = deepcopy(dataset1)
    dataset1.aug, dataset1.aug_ratio = args.aug1, args.aug_ratio1
    dataset2.aug, dataset2.aug_ratio = args.aug2, args.aug_ratio2

    loader1 = DataLoader(dataset1, batch_size=args.batch_size, num_workers = args.num_workers, shuffle=False)
    loader2 = DataLoader(dataset2, batch_size=args.batch_size, num_workers = args.num_workers, shuffle=False)

    model1.train()
    model2.train()

    train_loss_accum = 0

    for step, batch in enumerate(tqdm(zip(loader1, loader2), desc="Iteration")):
        batch1, batch2 = batch
        batch1 = batch1.to(device)
        batch2 = batch2.to(device)

        optimizer1.zero_grad()
        optimizer2.zero_grad()

        x1 = model1.forward_cl(batch1.x, batch1.edge_index, batch1.edge_attr, batch1.batch)
        x2 = model2.forward_cl(batch2.x, batch2.edge_index, batch2.edge_attr, batch2.batch)

        loss = model1.loss_cl(x1, x2, args.loss_temp, args)

        loss.backward()

        optimizer1.step()
        optimizer2.step()

        train_loss_accum += float(loss.detach().cpu().item())

        # 释放不必要的变量，防止显存泄漏
        x1 = None
        x2 = None
        loss = None
        torch.cuda.empty_cache()


    return train_loss_accum / (step + 1)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 80)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=1e-4,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=2,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=32,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0.1,
                        help='dropout ratio (default: 0)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--dataset', type=str, default='zinc_standard_agent',
                        help='root directory of dataset. For now, only classification.')
    parser.add_argument('--output_model_file', type=str, default='', help='filename to output the pre-trained model')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--seed', type=int, default=0, help="Random Seed")
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers for dataset loading')
    parser.add_argument('--aug1', type=str, default='dropN')
    parser.add_argument('--aug2', type=str, default='dropN')
    parser.add_argument('--aug_ratio1', type=float, default=0.2)
    parser.add_argument('--aug_ratio2', type=float, default=0.2)
    parser.add_argument('--kan_mlp', type = str, default='kan', help="mlp or kan")
    parser.add_argument('--kan_mp', type = str, default='none', help="kan or none")
    parser.add_argument('--kan_type1', type = str, default='ori', help="ori, bsrbf")
    parser.add_argument('--kan_type2', type = str, default='eff', help="ori, bsrbf")
    parser.add_argument('--grid', type = int, default = 5, help="bspline grid")
    parser.add_argument('--k', type = int, default = 3, help="bspline order")
    parser.add_argument('--neuron_fun', type = str, default = 'sum', help="kan's neuron_fun, in mean or sum")
    parser.add_argument('--use_transformer', type = str, default = 'mlp' , help="Use transformer: none, mlp or kan")
    parser.add_argument('--no_pos', action='store_true', help="Disable positive pair loss")
    parser.add_argument('--no_neg', action='store_true', help="Disable negative pair loss")

    pst = pytz.timezone('US/Pacific')
    current_time = datetime.now(pst).strftime('%Y%m%d_%H%M%S')
    parser.add_argument('--save_model_path', type=str, default=f"./models_rgcl/{current_time}/")
    parser.add_argument('--loss_temp', type=float, default=0.1)
    parser.add_argument('--input_model_file1', type=str, default="", help='filename to read the model (if there is any)')
    parser.add_argument('--input_model_file2', type=str, default="", help='filename to read the model (if there is any)')

    args = parser.parse_args()
    print(args)

    # Initialize wandb
    wandb.init(project="ICLR25_Trans", config=args)
    config = wandb.config

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    #set up dataset
    dataset = MoleculeDataset_aug("dataset/" + args.dataset, dataset=args.dataset)
    print(dataset)

    #set up model

    gnn1 = GNN(args.num_layer, args.emb_dim, JK=args.JK, drop_ratio=args.dropout_ratio, gnn_type=args.gnn_type,
              kan_mlp = args.kan_mlp, kan_mp = args.kan_mp, kan_type = args.kan_type1, grid = args.grid, k = args.k, 
              neuron_fun= args.neuron_fun, use_transformer = args.use_transformer)
    model1 = graphcl(gnn1, args)

    gnn2 = GNN(args.num_layer, args.emb_dim, JK=args.JK, drop_ratio=args.dropout_ratio, gnn_type=args.gnn_type,
              kan_mlp = args.kan_mlp, kan_mp = args.kan_mp, kan_type = args.kan_type2, grid = args.grid, k = args.k, 
              neuron_fun= args.neuron_fun, use_transformer = args.use_transformer)
    model2 = graphcl(gnn2, args)

    # 在初始化gnn1和gnn2之后，加载预训练模型
    if args.input_model_file1 != "":
        print("Loading pretrained model for gnn1 from", args.input_model_file1)
        gnn1.load_state_dict(torch.load(args.input_model_file1, map_location=device))

    if args.input_model_file2 != "":
        print("Loading pretrained model for gnn2 from", args.input_model_file2)
        gnn2.load_state_dict(torch.load(args.input_model_file2, map_location=device))

    model1.to(device)
    model2.to(device)

    summary(model1)
    summary(model2)

    #set up optimizer
    optimizer1 = optim.Adam(model1.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer2 = optim.Adam(model2.parameters(), lr=args.lr, weight_decay=args.decay)

    print(optimizer1)
    print(optimizer2)

    for epoch in tqdm(range(1, args.epochs + 1)):
        print("====epoch " + str(epoch))
    
        train_loss = train(args, model1, model2, device, dataset, optimizer1, optimizer2)

        print("train_loss: ", train_loss)

        # Log metrics to wandb
        wandb.log({
            f'train_loss_{args.dataset}': train_loss,
        })

        if epoch % 10 == 0:

            if not os.path.exists(args.save_model_path):
                os.makedirs(args.save_model_path)
                log_file_path = os.path.join(args.save_model_path, 'log.txt')
                with open(log_file_path, 'w') as log_file:
                    log_file.write(args.save_model_path)

            torch.save(gnn1.state_dict(), args.save_model_path + "rgcl_seed" +str(args.seed) + "_" + str(epoch) + "_gnn1.pth")
            torch.save(gnn2.state_dict(), args.save_model_path + "rgcl_seed" +str(args.seed) + "_" + str(epoch) + "_gnn2.pth")



if __name__ == "__main__":
    main()
