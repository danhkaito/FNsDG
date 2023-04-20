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
        num_neighbors=[10]*args.n_hop,
        batch_size=args.batch_size_gnn,
        input_nodes=homo_data_test.test_mask,
    )
# if arg.mode_sampling.lower() == 'dataloader':
#     train_loader = DataLoader(
#         homo_data_test,
#         batch_size=args.batch_size_gnn,
#     )
# if arg.mode_sampling.lower() == 'cluster':
#     cluster_data = ClusterData(homo_data_test, num_parts=som.size[0]*som.size[1], recursive=False)
#     train_loader = ClusterLoader(cluster_data, batch_size=1, shuffle=True,
#                                 num_workers=12)
# subgraph_loader = NeighborLoader(
#     data_test,
#     num_neighbors=[10, 5],
#     batch_size=32,
#     input_nodes=data_test.test_mask,
# )
# # for i, subgraph in enumerate(train_loader):
# #     print(f'Subgraph {i}: {subgraph}')
model_dict={
    "GCN": FNGCN(args.nhid1, args.nhid2, som.size[2], args.num_classes, args.n_hop, args.dropout, args.seed),
    "Sage": FNSage(args.nhid1, args.nhid2, som.size[2], args.num_classes, args.n_hop, args.dropout, args.seed),
    "RGCN": FNRGCN(args.nhid1, args.nhid2, som.size[2], args.num_classes, args.n_hop, args.dropout, args.seed),

}

FOLDER = f"../../saved/SOM-GNN/model/GNN/"
if not os.path.exists(FOLDER):
    os.makedirs(FOLDER)

model_fakenew = torch.load(f"{FOLDER}{args.train}/{args.embedding_model}/{args.gnn_model}_{args.graph_mode}.pt")

# model_fakenew = GIN(16)
print(model_fakenew)
# Use GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# features_content = features_content.to(device)
# features_style = features_style.to(device)
model_fakenew = model_fakenew.to(device)
# Initialize Optimizer

@torch.no_grad()
def test():
    predictions = []
    true_labels = []
    model_fakenew.eval()
    for batch in test_loader:
        batch = batch.to(device)
        out = model_fakenew(batch.x, batch.edge_index, batch.edge_type)
        batch_size = batch.batch_size
        output = out[:batch_size]
        b_labels = batch.y[:batch_size]
        output = output.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        
        # Store predictions and true labels
        predictions.append(np.argmax(output, axis=1).flatten())
        true_labels.append(label_ids.flatten())

    print('DONE.')
    pred_labels = (np.concatenate(predictions, axis=0))
    true_labels = (np.concatenate(true_labels, axis=0))
    
    print(f"Accuracy: {accuracy_score(true_labels, pred_labels)}\n \
          Precsion, Recall, F1-Score Label 1: {precision_recall_fscore_support(true_labels, pred_labels, average='binary', pos_label = 1)}\n\
          Precsion, Recall, F1-Score Label 0: {precision_recall_fscore_support(true_labels, pred_labels, average='binary', pos_label = 0)}")

      
test()

