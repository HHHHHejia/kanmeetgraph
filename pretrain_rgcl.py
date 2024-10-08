import argparse
from loader import MoleculeDataset_aug_rgcl
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
import torch_scatter
import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary

import numpy as np
from tqdm import tqdm
import os

from model import GNN, GNN_imp_estimator

from copy import deepcopy
import gc
import wandb
from datetime import datetime
import pytz



class graphcl(nn.Module):
    def __init__(self, gnn, node_imp_estimator, args):
        super(graphcl, self).__init__()
        self.gnn = gnn
        self.node_imp_estimator = node_imp_estimator
        self.pool = global_mean_pool
        self.emb_dim = args.emb_dim
        self.projection_head = nn.Sequential(nn.Linear(self.emb_dim, self.emb_dim), nn.ReLU(inplace=True), nn.Linear(self.emb_dim, self.emb_dim))

    def forward_cl(self, x, edge_index, edge_attr, batch):
        node_imp = self.node_imp_estimator(x, edge_index, edge_attr, batch)
        x = self.gnn(x, edge_index, edge_attr)

        out, _ = torch_scatter.scatter_max(torch.reshape(node_imp, (1, -1)), batch)
        out = out.reshape(-1, 1)
        out = out[batch]
        node_imp /= (out * 10)
        node_imp += 0.9
        node_imp = node_imp.expand(-1, self.emb_dim)

        x = torch.mul(x, node_imp)
        x = self.pool(x, batch)
        x = self.projection_head(x)
        return x

    def loss_cl(self, x1, x2, temp):
        T = temp
        batch_size, _ = x1.size()
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
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

    def loss_ra(self, x1, x2, x3, temp, lamda, beta):
        batch_size, _ = x1.size()

        # x feature: (batch, dim = 300)
        #R1(g), gen from R
        x1_abs = x1.norm(dim=1)
        #R2(g), gen from R
        x2_abs = x2.norm(dim=1)
        #R3(g), gen from R^c, env
        x3_abs = x3.norm(dim=1)

        #x1 x2 相似度矩阵，负样本为同batch中不同图的增强
        # sim_matrix feature: (batch, batch), range (0,1),很奇怪
        sim_matrix_raw = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        pos_sim_ori = sim_matrix_raw[range(batch_size), range(batch_size)]

        # sim_matrix feature: (batch, batch), range (1,e^(1/tmp)),很奇怪
        sim_matrix = torch.exp(sim_matrix_raw / temp)
        #x1 x2 正样本,  (batch)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]

        #loss su
        ra_loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        ra_loss = - torch.log(ra_loss).mean()

        #x1 x3 相似度矩阵，负样本为环境图
        # cp_sim_matrix feature: (batch, batch), range (0,1),很奇怪
        cp_sim_matrix = torch.einsum('ik,jk->ij', x1, x3) / torch.einsum('i,j->ij', x1_abs, x3_abs)
        # cp_sim_matrix feature: (batch, batch), range (1, e^(1/tmp)),很奇怪
        cp_sim_matrix = torch.exp(cp_sim_matrix / temp)

        #loss in
        cp_loss = pos_sim / (cp_sim_matrix.sum(dim=1) + pos_sim)
        cp_loss = - torch.log(cp_loss).mean()


        # 将 x1_aug_strength 和 x2_aug_strength 转换为 torch.Tensor 并移动到与 x1 相同的设备
        # x1_aug_strength = torch.tensor(x1_aug_strength, dtype=torch.float, device=x1.device)
        # x2_aug_strength = torch.tensor(x2_aug_strength, dtype=torch.float, device=x1.device)

        #pos sim: range (0,1), cal sim: range (0,1)
        # cal_sim_values = self.cal_sim(x1_aug_strength, x2_aug_strength)
        # cl_loss = torch.norm(pos_sim_ori - cal_sim_values, p = 2)/batch_size
        #cl_loss = 0
        loss = ra_loss + lamda * cp_loss
        return ra_loss, cp_loss, loss

    # def cal_sim(self, x1_aug_strength, x2_aug_strength):
    #     # 计算增强强度的差异
    #     strength_diff = torch.abs(x1_aug_strength - x2_aug_strength)
        
    #     # 线性插值计算相似度
    #     sim = 1- strength_diff
    #     return sim

def train(args, model, device, dataset, optimizer):

    dataset.aug = "none"
    imp_batch_size = 2048
    loader = DataLoader(dataset, batch_size=imp_batch_size, num_workers=args.num_workers, shuffle=False)
    model.eval()
    torch.set_grad_enabled(False)
    for step, batch in enumerate(loader):
        node_index_start = step*imp_batch_size
        node_index_end = min(node_index_start + imp_batch_size - 1, len(dataset)-1)
        batch = batch.to(device)
        node_imp = model.node_imp_estimator(batch.x, batch.edge_index, batch.edge_attr, batch.batch).detach()
        dataset.node_score[dataset.slices['x'][node_index_start]:dataset.slices['x'][node_index_end + 1]] = torch.squeeze(node_imp.half())

    dataset1 = deepcopy(dataset)
    dataset1 = dataset1.shuffle()
    dataset2 = deepcopy(dataset1)
    dataset3 = deepcopy(dataset1)

    dataset1.aug, dataset1.aug_ratio = args.aug1, args.aug_ratio1
    dataset2.aug, dataset2.aug_ratio = args.aug2, args.aug_ratio2
    dataset3.aug, dataset3.aug_ratio = args.aug2 + '_cp', args.aug_ratio1

    loader1 = DataLoader(dataset1, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    loader2 = DataLoader(dataset2, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    loader3 = DataLoader(dataset3, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    torch.set_grad_enabled(True)
    model.train()

    train_loss_accum = 0
    ra_loss_accum = 0
    cp_loss_accum = 0
    cl_loss_accum = 0

    for step, batch in enumerate(zip(loader1, loader2, loader3)):
        batch1, batch2, batch3 = batch
        batch1 = batch1.to(device)
        batch2 = batch2.to(device)
        batch3 = batch3.to(device)

        optimizer.zero_grad()

        x1 = model.forward_cl(batch1.x, batch1.edge_index, batch1.edge_attr, batch1.batch)
        x2 = model.forward_cl(batch2.x, batch2.edge_index, batch2.edge_attr, batch2.batch)
        x3 = model.forward_cl(batch3.x, batch3.edge_index, batch3.edge_attr, batch3.batch)

        # 获取增强强度

        ra_loss, cp_loss, loss = model.loss_ra(x1, x2, x3, args.loss_temp, args.lamda, args.beta)

        loss.backward()
        optimizer.step()

        train_loss_accum += float(loss.detach().cpu().item())
        ra_loss_accum += float(ra_loss.detach().cpu().item())
        cp_loss_accum += float(cp_loss.detach().cpu().item())
        #cl_loss_accum += float(cl_loss.detach().cpu().item())

    del dataset1, dataset2, dataset3
    gc.collect()
    return train_loss_accum/(step+1), ra_loss_accum/(step+1), cp_loss_accum/(step+1)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 80)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=30,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0,
                        help='dropout ratio (default: 0)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--dataset', type=str, default='zinc_standard_agent',
                        help='root directory of dataset. For now, only classification.')
    parser.add_argument('--output_model_file', type=str, default='', help='filename to output the pre-trained model')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--seed', type=int, default=0, help="Random Seed")
    parser.add_argument('--num_workers', type=int, default=32, help='number of workers for dataset loading')
    parser.add_argument('--aug1', type=str, default='dropN')
    parser.add_argument('--aug2', type=str, default='dropN')
    parser.add_argument('--aug_ratio1', type=float, default=0.2)
    parser.add_argument('--aug_ratio2', type=float, default=0.2)

    pst = pytz.timezone('US/Pacific')
    current_time = datetime.now(pst).strftime('%Y%m%d_%H%M%S')
    parser.add_argument('--save_model_path', type=str, default=f"./models_rgcl/{current_time}/")

    parser.add_argument('--loss_temp', type=float, default=0.1)
    parser.add_argument('--lamda', type=float, default=0.1)
    parser.add_argument('--beta', type=float, default=0)
    parser.add_argument(
        '--kan_mlp', 
        type = str,
        default='mlp', 
        help="mlp or kan"
    )
    parser.add_argument(
        '--kan_up', 
        type = str,
        default='none', 
        help="none, mlp or kan"
    )

    args = parser.parse_args()
    print(args)

    # Initialize wandb
    wandb.init(project="KAGNN", config=args)
    config = wandb.config

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    #set up dataset
    dataset = MoleculeDataset_aug_rgcl("dataset/" + args.dataset, dataset=args.dataset)
    print(dataset)

    #set up model
    gnn = GNN(args.num_layer, args.emb_dim, JK=args.JK, drop_ratio=args.dropout_ratio, gnn_type=args.gnn_type,
              kan_mlp=args.kan_mlp, kan_up = args.kan_up)
    node_imp_estimator = GNN_imp_estimator(num_layer=3, emb_dim=args.emb_dim, JK=args.JK, drop_ratio=args.dropout_ratio)

    model = graphcl(gnn, node_imp_estimator, args)
    
    model.to(device)
    summary(model)

    #set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    print(optimizer)

    for epoch in tqdm(range(1, args.epochs + 1)):
        print("====epoch " + str(epoch))
    
        train_loss, ra_loss, cp_loss = train(args, model, device, dataset, optimizer)

        print("train_loss: ", train_loss)
        print("ra_loss: ", ra_loss)
        print("cp_loss: ", cp_loss)
        #print("cl_loss: ", cl_loss)

        # Log metrics to wandb
        wandb.log({
            f'train_loss_{args.dataset}': train_loss,
            f'ra_loss_{args.dataset}': ra_loss,
            f'cp_loss_{args.dataset}': cp_loss,
            #f'cl_loss_{args.dataset}': cl_loss,
        })

        if epoch % 10 == 0:

            if not os.path.exists(args.save_model_path):
                os.makedirs(args.save_model_path)
                log_file_path = os.path.join(args.save_model_path, 'log.txt')
                with open(log_file_path, 'w') as log_file:
                    log_file.write(args.save_model_path)

            torch.save(gnn.state_dict(), args.save_model_path + "rgcl_seed" +str(args.seed) + "_" + str(epoch) + "_gnn.pth")
            torch.save(node_imp_estimator.state_dict(), args.save_model_path + "rgcl_seed" + str(args.seed) + "_" + str(epoch) +
                       "_rationale_generator.pth")


if __name__ == "__main__":
    main()
