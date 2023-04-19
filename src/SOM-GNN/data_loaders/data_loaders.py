import numpy as np
import math
from torch.utils.data import DataLoader, TensorDataset, SequentialSampler, RandomSampler
import numpy as np
import torch
from tqdm import tqdm
from scipy.spatial.distance import pdist
import networkx as nx

def create_edge_subgraph(som, train_X, graph_dict, node_idxs, prop = 0.75):

    length = graph_dict[node_idxs].shape[0]
    # print(length)
    map_node_idx = dict()
    node_arr = np.empty((graph_dict[node_idxs].shape[0], som.size[2]))
    # print(node_arr.shape)
    for i in range (graph_dict[node_idxs].shape[0]):
        map_node_idx[i] = graph_dict[node_idxs][i]
        if graph_dict[node_idxs][i] - som.size[0]*som.size[1] > train_X.shape[0]-1:
            print("!nát!")
        else:
            node_arr[i] = train_X[graph_dict[node_idxs][i] - som.size[0]*som.size[1]]
        

    dist = pdist(node_arr, metric='cosine')
    dist = 1 - dist
    num_edge = length*(length-1)/2
    top_k = int(prop*num_edge)
    top_idx = np.argpartition(dist, -top_k)[-top_k:]
    row,col = np.triu_indices(length,k=1)
    lst_edge = np.concatenate((row[top_idx].reshape(-1,1), col[top_idx].reshape(-1,1)), axis=1)
    if lst_edge.shape[0]==0:
        print("Not have any edge")
        # print(top_idx)
        return np.array([]), np.empty((0,3))
    mapping_fn = lambda x: map_node_idx[x]
    vec_mapfn = np.vectorize(mapping_fn)
    lst_edge = vec_mapfn(lst_edge)
    edge_weight = dist[top_idx]
    edge_att = lambda x: {'edge_weight':x,'edge_type':1}
    vec_edge_att = np.vectorize(edge_att)
    lst_edge_att = vec_edge_att(edge_weight).reshape(-1, 1)
    lst_edge = np.concatenate((lst_edge, lst_edge_att), axis=1)
    return dist, lst_edge

def create_whole_Graph(dataset, train_X, train_Y, som, som_connected = False, connected_radius = 1, batch_size = 512):
    train_X_torch = torch.from_numpy(train_X)
    train_Y_torch = torch.from_numpy(train_Y)
    sampler_test = SequentialSampler(data_source=dataset)
    data_test = DataLoader(dataset, sampler=sampler_test, batch_size=batch_size)
    graph_dict = dict()
    flatten_weights = som.weights.view(som.size[0]*som.size[1], -1).cpu()
    node_list = np.empty((0,2))
    edge_list = np.empty((0,3))


    for i in range(som.size[0]*som.size[1]):
        node_list = np.append(node_list, np.array([i, {'x': flatten_weights[i], 'y': 2, 'train_mask': False, 'test_mask':False}]).reshape(1, -1), axis= 0)
    
    if som_connected:
        for i in range(som.size[0]*som.size[1]):
            index_rad = (np.transpose(np.stack(np.ones((2*connected_radius+1, 2*connected_radius+1)).nonzero()))-connected_radius)
            zero_idx = (index_rad == np.array([0,0])).astype(np.int32)
            selected = np.where(zero_idx[:,0] + zero_idx[:,1] == 1)[0]
            _2d_idx = np.array([i//som.size[0], i%som.size[1]])
            selected_neighbor = _2d_idx + index_rad[selected]
            greater_zero_idx = selected_neighbor >= np.array([0,0])
            less_rad = selected_neighbor < np.array([som.size[0], som.size[1]])
            gre_les = np.logical_and(greater_zero_idx, less_rad)
            selected_greaterzero = np.where(np.logical_and(gre_les[:,0], gre_les[:,1]))[0]
            selected_neighbor = selected_neighbor[selected_greaterzero]
            edge_somblock = np.array([[i, sn[0]*som.size[0] + sn[1], {'edge_weight': pdist([flatten_weights[i].numpy(), flatten_weights[sn[0]*som.size[0] + sn[1]].numpy()])[0],'edge_type':2}] for sn in selected_neighbor])
            edge_list = np.concatenate((edge_list, edge_somblock))
                    
    for x in som.map_indices.numpy():
        graph_dict[tuple(x)] = np.array([], dtype=np.int32)
    for idx, batch in enumerate(data_test):
        batch = batch[0].to(device='cuda:0')
        bmus = som(batch)
        bmus_np = bmus.cpu().numpy()
        map_indices = som.map_indices.cpu().numpy()
        for i in range (bmus_np.shape[0]):
            node_list = np.append(node_list, np.array([i+idx*batch_size+som.size[0]*som.size[1], {'x': train_X_torch[i+idx*batch_size],'y': train_Y_torch[i+idx*batch_size], 'train_mask': True, 'test_mask':False}]).reshape(1,-1), axis= 0)
        for x in map_indices:

            t = (bmus_np==x)
            y = np.where(np.logical_and(t[:,0], t[:,1]))[0] + idx*batch_size +som.size[0]*som.size[1]
            # print(y)
            if (len(y)==0):
                continue
            graph_dict[tuple(x)] = np.concatenate((graph_dict[tuple(x)], y))
            if som_connected:
                _1d_index =  x[0]*som.size[0] + x[1]
                edge_som = np.array([[_1d_index, y[i], {'edge_weight': 1 - pdist([flatten_weights[_1d_index].numpy(), train_X[y[i]-som.size[0]*som.size[1]]], metric='cosine')[0],'edge_type':0}] for i in range(y.shape[0])])
                edge_list = np.concatenate((edge_list, edge_som))
    for x in graph_dict:
        _, adj = create_edge_subgraph(graph_dict, x)
        edge_list = np.concatenate((edge_list, adj))
    return node_list, edge_list, graph_dict

def create_homogeneous_data(name_train, name_test = None):
    train_Y = np.load(f"../clean data/{name_train}/data_embedding/bert-base-cased/train_label_128.npy")
    train_X = np.load(f"../clean data/{name_test}/data_embedding/bert-base-cased/train_embedding_128.npy")

    X_train = 
    if path_test:
        train_X_test = 
