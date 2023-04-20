import sys
import os

import random
import pandas as pd
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import ClusterLoader, ClusterData, NeighborLoader, DataLoader
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support

from sklearn.model_selection import train_test_split
import torch_geometric.transforms as T
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from conf.parse_config import get_parser, parse_args
from model.GNN.gnn import *
from data_loaders.data_loaders import create_homogeneous_data
from tqdm import tqdm


parser = get_parser()
arg = parser.parse_args()

args = parse_args(parser, arg.conf)
args = args[0]

set_seed(args.seed)

som = torch.load(f"../../saved/SOM-GNN/model/SOM/{args.train}/{args.embedding_model}/som_{args.epoch_som}_{args.batch_size_som}.pt")

homo_data = create_homogeneous_data(som, args.embedding_model, args.train, args.som_connected == "True", preload=args.preload_data_train=="True")


# print(homo_data)
# print(np.unique(homo_data.edge_type.numpy()))

if args.mode_sampling.lower() == 'neighbor':
    train_loader = NeighborLoader(
        homo_data,
        num_neighbors=[10]*args.n_hop,
        batch_size=args.batch_size_gnn,
        input_nodes=homo_data.train_mask,
    )
if args.mode_sampling.lower() == 'dataloader':
    train_loader = DataLoader(
        homo_data,
        batch_size=args.batch_size_gnn,
    )
if args.mode_sampling.lower() == 'cluster':
    cluster_data = ClusterData(homo_data, num_parts=som.size[0]*som.size[1], recursive=False)
    train_loader = ClusterLoader(cluster_data, batch_size=1, shuffle=True,
                                num_workers=12)
# subgraph_loader = NeighborLoader(
#     data_test,
#     num_neighbors=[10, 5],
#     batch_size=32,
#     input_nodes=data_test.test_mask,
# )
# # for i, subgraph in enumerate(train_loader):
# #     print(f'Subgraph {i}: {subgraph}')
homo_data_test = create_homogeneous_data(som, args.embedding_model, args.train, args.som_connected == "True", preload=args.preload_data_test=="True", name_test=args.test)

print(homo_data_test)

print(np.unique(homo_data_test.edge_type.numpy()))
# fake_idx = np.squeeze(np.argwhere(data.y == 1))
# X_fake_train, X_fake_test = train_test_split(fake_idx, test_size=0.2, random_state=42)
# true_idx = np.squeeze(np.argwhere(data.y == 0))
# X_true_train, X_true_test = train_test_split(true_idx, test_size=1- len(X_fake_train)/len(true_idx), random_state=42)
# X_train=np.concatenate((true_idx,fake_idx), axis=None)
# X_test=np.concatenate((X_true_test,X_fake_test), axis=None)
# X_train=np.concatenate((X_train, X_test), axis=None)


# print(len(X_fake_train), len(X_true_train))
# a = np.squeeze(np.argwhere(X_train < 15*15))
# print(len(a))
# exit()
############# Train mask ############
# Mask all the ksom node
# train_mask = torch.zeros(len(data.x), dtype=torch.bool)
# test_mask = torch.zeros(len(data.x), dtype=torch.bool)
# train_mask[X_train] = True
# test_mask[X_test] = True
# for x in range(0, 15*15):
#     test_mask[x]=True


# for i in range (0, 15*15):
#     train_mask[i] = False
if args.mode_sampling.lower() == 'neighbor':
    test_loader = NeighborLoader(
        homo_data_test,
        num_neighbors=[-1]*args.n_hop,
        batch_size=args.batch_size_gnn,
        input_nodes=homo_data_test.test_mask,
        shuffle = False,
    )
    
model_dict={
    "GCN": FNGCN(args.nhid1, args.nhid2, som.size[2], args.num_classes, args.n_hop, args.dropout, args.seed),
    "Sage": FNSage(args.nhid1, args.nhid2, som.size[2], args.num_classes, args.n_hop, args.dropout, args.seed),
    "RGCN": FNRGCN(args.nhid1, args.nhid2, som.size[2], args.num_classes, args.n_hop, args.dropout, args.seed),

}
model_fakenew = model_dict[args.gnn_model]
# model_fakenew = GIN(16)
print(model_fakenew)
# Use GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# features_content = features_content.to(device)
# features_style = features_style.to(device)
model_fakenew = model_fakenew.to(device)
# Initialize Optimizer
learning_rate = args.lr
decay = args.decay

optimizer = torch.optim.Adam(model_fakenew.parameters(), 
                             lr=learning_rate, 
                             weight_decay= decay)


# Define loss function (CrossEntropyLoss for Classification Problems with 
# probability distributions)
criterion = torch.nn.CrossEntropyLoss()

def train():
    train_loss = 0
    for batch in tqdm(train_loader):
        model_fakenew.train()
        optimizer.zero_grad()
        # Use all data as input, because all nodes have node features
        batch_size = batch.batch_size
        batch = batch.to(device)
        out = model_fakenew(batch.x, batch.edge_index, batch.edge_type) 
    #   print_class_acc(out[data.train_mask], data.y[data.train_mask])

        # Only use nodes with labels available for loss calculation --> mask
        loss = criterion(out[:batch_size], batch.y[:batch_size])  
        loss.backward()
        optimizer.step()
        train_loss += loss

    return train_loss/len(train_loader)
@torch.no_grad()
def test():
    labels = np.array([])
    outs = torch.empty((0,2))
    model_fakenew.eval()
    for batch in test_loader:
        batch = batch.to(device)
        out = model_fakenew(batch.x, batch.edge_index, batch.edge_type)
        batch_size = batch.batch_size
        output = out[:batch_size]
        b_labels = batch.y[:batch_size]
        output = output.detach().cpu()
        label_ids = b_labels.cpu().numpy()
        # Store predictions and true labels
        labels = np.concatenate((labels, label_ids))
        outs = torch.cat((outs, output))
    print_class_acc(outs, labels)

      

losses = []
for epoch in range(0, args.epoch_gnn+1):
    loss = train()
    losses.append(loss)
    if epoch % 5 == 0:
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
    # if epoch == args.epoch_gnn:
        test()

FOLDER = f"../../saved/SOM-GNN/model/GNN/{args.train}/{args.embedding_model}/"
if not os.path.exists(FOLDER):
    os.makedirs(FOLDER)
connected = "Connected" if args.som_connected=="True" else "UnConnected"

torch.save(model_fakenew, f"{FOLDER}{args.gnn_model}_{args.graph_mode}_{connected}.pt")
# model_fakenew = torch.load(f"{FOLDER}{args.gnn_model}_{args.graph_mode}_{connected}.pt")

# # model_fakenew = GIN(16)
# print(model_fakenew)
# # Use GPU
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# # features_content = features_content.to(device)
# # features_style = features_style.to(device)
# model_fakenew = model_fakenew.to(device)
# # Initialize Optimizer

      
test()

