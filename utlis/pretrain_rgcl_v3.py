import argparse
from loader import MoleculeDataset_aug_rgcl
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
import torch_scatter
import torch
import torch.nn as nn
import torch.optim as optim
import time

import numpy as np
from tqdm import tqdm
import os

from model import GNN, GNN_imp_estimator

from copy import deepcopy
import gc
import wandb
from datetime import datetime
import pytz

from torch.distributions import Normal


class graphcl(nn.Module):
    def __init__(self, gnn, node_imp_estimator):
        super(graphcl, self).__init__()
        self.gnn = gnn
        self.node_imp_estimator = node_imp_estimator
        self.pool = global_mean_pool
        self.projection_head = nn.Sequential(nn.Linear(300, 300), nn.ReLU(inplace=True), nn.Linear(300, 300))

    def forward_cl(self, x, edge_index, edge_attr, batch):
        node_imp = self.node_imp_estimator(x, edge_index, edge_attr, batch)
        x = self.gnn(x, edge_index, edge_attr)

        out, _ = torch_scatter.scatter_max(torch.reshape(node_imp, (1, -1)), batch)
        out = out.reshape(-1, 1)
        out = out[batch]
        node_imp /= (out * 10)
        node_imp += 0.9
        node_imp = node_imp.expand(-1, 300)
        
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

    def loss_ra(self, x1, x2, x4, 
                temp, gamma, 
                graph_dis):
        batch_size, _ = x1.size()

        # x feature: (batch, dim = 300)
        #R1(g), gen from R
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)
        #1.标准constrastive loss： x1，x2
        # x1 x2 相似度矩阵，负样本为同batch中不同图的增强
        # sim_matrix feature: (batch, batch), range (0,1),很奇怪
        sim_matrix_raw = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        # sim_matrix feature: (batch, batch), range (1,e^(1/tmp)),很奇怪
        sim_matrix = torch.exp(sim_matrix_raw / temp)
        #x1 x2 正样本,  (batch)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        neg_sim = sim_matrix.sum(dim=1) - pos_sim
        #loss su
        ra_loss = pos_sim / neg_sim
        ra_loss = - torch.log(ra_loss).mean()
        #3.iso loss： x1，x4
        #pos sim: range (0,1), cal sim: range (0,1)

        feature_sim = torch.sum(x1 / x1.norm(dim=1, keepdim=True) * x4 / x4.norm(dim=1, keepdim=True), dim=1)  

        #D(f(x1,x4))
        feature_dis = self.cal_fea_dis(feature_sim)
        #d(x1, x4)
        cl_loss = torch.norm(feature_dis - graph_dis, p = 2)

        loss = ra_loss + gamma * cl_loss
        return ra_loss, cl_loss, loss

    def cal_fea_dis(self, feature_sim):        
        # Apply clamp to ensure values are within a safe range for acos
        epsilon = 1e-7
        feature_sim = torch.clamp(feature_sim, -1 + epsilon, 1 - epsilon)
        geo_distance = (2 / torch.pi) * torch.acos(feature_sim)
        return geo_distance
     
    def cal_graph_dis(self, node_importance, x1_idx_nondrop, x4_idx_nondrop, device):

        # 关闭梯度计算
        with torch.no_grad():
            max_len = max(map(len, node_importance))
            importance_tensor = torch.zeros((len(node_importance), max_len), device=device)

            for i, imp in enumerate(node_importance):
                importance_tensor[i, :len(imp)] = torch.tensor(imp, device=device)

            # 填充时直接将无效值设置为最大合法索引
            max_len_nondrop1 = max(map(len, x1_idx_nondrop))
            max_len_nondrop2 = max(map(len, x4_idx_nondrop))
            max_len_nondrop = max(max_len_nondrop1, max_len_nondrop2)

            idx_nondrop1_tensor = torch.full((len(x1_idx_nondrop), max_len_nondrop), max_len, device=device, dtype=torch.long)
            idx_nondrop2_tensor = torch.full((len(x4_idx_nondrop), max_len_nondrop), max_len, device=device, dtype=torch.long)
            

            for i, (idx1, idx2) in enumerate(zip(x1_idx_nondrop, x4_idx_nondrop)):
                idx_nondrop1_tensor[i, :len(idx1)] = torch.tensor(idx1, device=device)
                idx_nondrop2_tensor[i, :len(idx2)] = torch.tensor(idx2, device=device)


            # 创建布尔掩码
            mask1 = torch.ones_like(importance_tensor, dtype=torch.bool, device=device)
            mask2 = torch.ones_like(importance_tensor, dtype=torch.bool, device=device)

            # 使用 scatter_add 处理掩码
            mask1.scatter_(1, idx_nondrop1_tensor.clamp(max=max_len-1), False)
            mask2.scatter_(1, idx_nondrop2_tensor.clamp(max=max_len-1), False)

            # 计算对称差
            sym_diff_mask = mask1 ^ mask2

            # 计算对称差中节点的重要性和
            sym_diff_importance_sum = torch.sum(importance_tensor * sym_diff_mask, dim=1)

            # 计算参考节点（所有节点）的重要性和
            anchor_importance_sum = torch.sum(importance_tensor, dim=1)

            # 计算图距离
            graph_distances = sym_diff_importance_sum / anchor_importance_sum
        # 返回张量
        return graph_distances

def train(args, model, device, dataset, optimizer, low, high):

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
    
    dataset = dataset.shuffle()
    dataset1 = deepcopy(dataset)
    dataset2 = deepcopy(dataset)
    dataset4 = deepcopy(dataset)

    # 使用 epsilon 进行数据增强
    dataset1.aug, dataset1.aug_ratio = args.aug1, args.aug_ratio1
    dataset2.aug, dataset2.aug_ratio = args.aug2, args.aug_ratio2
    dataset4.aug, dataset4.low, dataset4.high  = args.aug4, low, high

    loader1 = DataLoader(dataset1, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    loader2 = DataLoader(dataset2, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    loader4 = DataLoader(dataset4, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)


    torch.set_grad_enabled(True)
    model.train()

    train_loss_accum = 0
    ra_loss_accum = 0
    cl_loss_accum = 0

    #特殊情况

    for step, batch in enumerate(zip(loader1, loader2, loader4)):
    
        batch1, batch2, batch4 = batch
        batch1 = batch1.to(device)
        batch2 = batch2.to(device)
        batch4 = batch4.to(device)

        optimizer.zero_grad()

        x1 = model.forward_cl(batch1.x, batch1.edge_index, batch1.edge_attr, batch1.batch)
        x2 = model.forward_cl(batch2.x, batch2.edge_index, batch2.edge_attr, batch2.batch)
        x4 = model.forward_cl(batch4.x, batch4.edge_index, batch4.edge_attr, batch4.batch)


        # 获取增强强度

        #graph_dis = model.cal_graph_dis(batch1.node_prob, batch1.idx_nondrop, batch4.idx_nondrop, device)
        # 将列表转换为张量
        batch1_strength_tensor = torch.tensor(batch1.augmentation_strength, dtype=torch.float32, requires_grad = False).to(device)
        batch4_strength_tensor = torch.tensor(batch4.augmentation_strength, dtype=torch.float32, requires_grad = False).to(device)
        graph_dis = torch.abs(batch1_strength_tensor - batch4_strength_tensor).to(device)

        ra_loss, cl_loss, loss = model.loss_ra(x1, x2, x4, 
                                                args.loss_temp, 
                                                args.gamma, 
                                                graph_dis)


        loss.backward()
        optimizer.step()


        train_loss_accum += float(loss.detach().cpu().item())
        ra_loss_accum += float(ra_loss.detach().cpu().item())
        cl_loss_accum += float(cl_loss.detach().cpu().item())

    del dataset1, dataset2, dataset4
    gc.collect()
    return train_loss_accum/(step+1), ra_loss_accum/(step+1), cl_loss_accum/(step+1)


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
    parser.add_argument('--emb_dim', type=int, default=300,
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
    parser.add_argument('--aug4', type=str, default='dropN_random')

    parser.add_argument('--aug_ratio1', type=float, default=0.2)
    parser.add_argument('--aug_ratio2', type=float, default=0.2)

    pst = pytz.timezone('US/Pacific')
    current_time = datetime.now(pst).strftime('%Y%m%d_%H%M%S')
    parser.add_argument('--save_model_path', type=str, default=f"./models_rgcl/{current_time}/")

    parser.add_argument('--loss_temp', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0)

    parser.add_argument('--gnn_ckpt', type=str, default="")
    #default:  ./models_rgcl/no36/rgcl_seed0_100_gnn.pth
    parser.add_argument('--rational_ckpt', type=str, default="")
    #default: ./models_rgcl/no36/rgcl_seed0_100_rationale_generator.pth

    parser.add_argument('--low', type=float, default=0)
    parser.add_argument('--high', type=float, default=0.4)

    args = parser.parse_args()
    print(args)


    # Initialize wandb
    wandb.init(project="GNNCL", config=args)
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
    gnn = GNN(args.num_layer, args.emb_dim, JK=args.JK, drop_ratio=args.dropout_ratio, gnn_type=args.gnn_type)
    node_imp_estimator = GNN_imp_estimator(num_layer=3, emb_dim=args.emb_dim, JK=args.JK, drop_ratio=args.dropout_ratio)

    # Load checkpoints
    if (args.gnn_ckpt and args.rational_ckpt):
        print("loading gnn:", args.gnn_ckpt)
        gnn.load_state_dict(torch.load(args.gnn_ckpt, map_location=device))
        print("loading gnn:", args.rational_ckpt)
        node_imp_estimator.load_state_dict(torch.load(args.rational_ckpt, map_location=device))

    model = graphcl(gnn, node_imp_estimator) 
    
    model.to(device)

    # 初始化均值和标准差
    # mu = nn.Parameter(torch.tensor(args.mu))  # 初始值可以根据经验设定
    # sigma = nn.Parameter(torch.tensor(args.sigma))  # 初始值可以根据经验设定
    #set up optimizer
    optimizer = optim.Adam(list(model.parameters()), lr=args.lr, weight_decay=args.decay)
    print(optimizer)
    for epoch in tqdm(range(1, args.epochs + 1)):
        print("====epoch " + str(epoch))
    
        train_loss, ra_loss, cl_loss = train(args, model, device, dataset, optimizer,args.low, args.high)

        print("train_loss: ", train_loss)
        print("ra_loss: ", ra_loss)
        print("cl_loss: ", cl_loss)

        # Log metrics to wandb
        wandb.log({
            f'train_loss_{args.dataset}': train_loss,
            f'ra_loss_{args.dataset}': ra_loss,
            f'cl_loss_{args.dataset}': cl_loss,
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
