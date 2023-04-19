import numpy as np
import math
from torch.utils.data import DataLoader, TensorDataset, SequentialSampler, RandomSampler
import numpy as np
import torch
from tqdm import tqdm
from scipy.spatial.distance import pdist
import networkx as nx
from typing import List, Optional, Tuple, Union
from collections import defaultdict
from torch import Tensor
import os
def create_edge_subgraph(som, X, graph_dict, node_idxs, prop = 0.75):

    length = graph_dict[node_idxs].shape[0]
    # print(length)
    map_node_idx = dict()
    node_arr = np.empty((graph_dict[node_idxs].shape[0], som.size[2]))
    # print(node_arr.shape)
    for i in range (graph_dict[node_idxs].shape[0]):
        map_node_idx[i] = graph_dict[node_idxs][i]
        if graph_dict[node_idxs][i] - som.size[0]*som.size[1] > X.shape[0]-1:
            print("!nát!")
        else:
            node_arr[i] = X[graph_dict[node_idxs][i] - som.size[0]*som.size[1]]
        

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

def create_whole_Graph(X, Y, som, som_connected = False, connected_radius = 1, batch_size = 512, len_train = math.inf):
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
    X_torch = torch.from_numpy(X)
    Y_torch = torch.from_numpy(Y)
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
            real_idx = i+idx*batch_size
            if real_idx < len_train:
                node_list = np.append(node_list, np.array([real_idx+som.size[0]*som.size[1], {'x': X_torch[real_idx],'y': Y_torch[real_idx], 'train_mask': True, 'test_mask':False}]).reshape(1,-1), axis= 0)
            else:
                # print("Have test")
                node_list = np.append(node_list, np.array([real_idx+som.size[0]*som.size[1], {'x': X_torch[real_idx],'y': Y_torch[real_idx], 'train_mask': False, 'test_mask':True}]).reshape(1,-1), axis= 0)

        for x in map_indices:

            t = (bmus_np==x)
            y = np.where(np.logical_and(t[:,0], t[:,1]))[0] + idx*batch_size +som.size[0]*som.size[1]
            # print(y)
            if (len(y)==0):
                continue
            graph_dict[tuple(x)] = np.concatenate((graph_dict[tuple(x)], y))
            if som_connected:
                _1d_index =  x[0]*som.size[0] + x[1]
                edge_som = np.array([[_1d_index, y[i], {'edge_weight': 1 - pdist([flatten_weights[_1d_index].numpy(), X[y[i]-som.size[0]*som.size[1]]], metric='cosine')[0],'edge_type':0}] for i in range(y.shape[0])])
                edge_list = np.concatenate((edge_list, edge_som))
    for x in graph_dict:
        _, adj = create_edge_subgraph(som, X, graph_dict, x)
        edge_list = np.concatenate((edge_list, adj))
    return node_list, edge_list, graph_dict


def from_networkx(G, group_node_attrs: Optional[Union[List[str], all]] = None,
                  group_edge_attrs: Optional[Union[List[str], all]] = None):
    r"""Converts a :obj:`networkx.Graph` or :obj:`networkx.DiGraph` to a
    :class:`torch_geometric.data.Data` instance.

    Args:
        G (networkx.Graph or networkx.DiGraph): A networkx graph.
        group_node_attrs (List[str] or all, optional): The node attributes to
            be concatenated and added to :obj:`data.x`. (default: :obj:`None`)
        group_edge_attrs (List[str] or all, optional): The edge attributes to
            be concatenated and added to :obj:`data.edge_attr`.
            (default: :obj:`None`)

    .. note::

        All :attr:`group_node_attrs` and :attr:`group_edge_attrs` values must
        be numeric.
    """
    import networkx as nx

    from torch_geometric.data import Data

    G = nx.convert_node_labels_to_integers(G)
    # G = G.to_directed() if not nx.is_directed(G) else G

    if isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)):
        edges = list(G.edges(keys=False))
    else:
        edges = list(G.edges)

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    data = defaultdict(list)

    if G.number_of_nodes() > 0:
        node_attrs = list(next(iter(G.nodes(data=True)))[-1].keys())
    else:
        node_attrs = {}

    if G.number_of_edges() > 0:
        edge_attrs = list(next(iter(G.edges(data=True)))[-1].keys())
    else:
        edge_attrs = {}

    for i, (_, feat_dict) in enumerate(G.nodes(data=True)):
        if set(feat_dict.keys()) != set(node_attrs):
            raise ValueError('Not all nodes contain the same attributes')
        for key, value in feat_dict.items():
            data[str(key)].append(value)

    for i, (_, _, feat_dict) in enumerate(G.edges(data=True)):
        if set(feat_dict.keys()) != set(edge_attrs):
            raise ValueError('Not all edges contain the same attributes')
        for key, value in feat_dict.items():
            key = f'edge_{key}' if key in node_attrs else key
            data[str(key)].append(value)

    for key, value in G.graph.items():
        key = f'graph_{key}' if key in node_attrs else key
        data[str(key)] = value

    for key, value in data.items():
        if isinstance(value, (tuple, list)) and isinstance(value[0], Tensor):
            data[key] = torch.stack(value, dim=0)
        else:
            try:
                data[key] = torch.tensor(value)
            except ValueError:
                pass

    data['edge_index'] = edge_index.view(2, -1)
    data = Data.from_dict(data)

    if group_node_attrs is all:
        group_node_attrs = list(node_attrs)
    if group_node_attrs is not None:
        xs = []
        for key in group_node_attrs:
            x = data[key]
            x = x.view(-1, 1) if x.dim() <= 1 else x
            xs.append(x)
            del data[key]
        data.x = torch.cat(xs, dim=-1)

    if group_edge_attrs is all:
        group_edge_attrs = list(edge_attrs)
    if group_edge_attrs is not None:
        xs = []
        for key in group_edge_attrs:
            key = f'edge_{key}' if key in node_attrs else key
            x = data[key]
            x = x.view(-1, 1) if x.dim() <= 1 else x
            xs.append(x)
            del data[key]
        data.edge_attr = torch.cat(xs, dim=-1)

    if data.x is None and data.pos is None:
        data.num_nodes = G.number_of_nodes()

    return data

def create_homogeneous_data(som, embeddings, name_train, som_connected, preload = False, name_test = None):
    FOLDER = f"../../saved/SOM-GNN/data/{name_train}/{embeddings}"
    if not os.path.exists(FOLDER):
        os.makedirs(FOLDER)

    if name_test:
        FOLDER = f"../../saved/SOM-GNN/data/{name_train}+{name_test}/{embeddings}"
        if not os.path.exists(FOLDER):
            os.makedirs(FOLDER)


    if preload == False:
        X = np.load(f"../../saved/embeddings/{name_train}/{embeddings}/train_embedding.npy")
        Y = np.load(f"../../saved/embeddings/{name_train}/{embeddings}/train_label.npy")
        len_train = math.inf
        if name_test is not None:
            X_test = np.load(f"../../saved/embeddings/{name_test}/{embeddings}/test_embedding.npy")
            Y_test = np.load(f"../../saved/embeddings/{name_test}/{embeddings}/test_label.npy")
            len_train = len(X)
            X = np.concatenate((X, X_test))
            Y = np.concatenate((Y, Y_test))
        node_list, edge_list, mapping_som = create_whole_Graph(X, Y, som, som_connected=som_connected, len_train=len_train)
        G = nx.Graph()
        G.add_nodes_from(node_list)
        G.add_edges_from(edge_list)
        pyg_data = from_networkx(G)
        edge_inv = torch.cat((pyg_data.edge_index[1].view(1,-1), pyg_data.edge_index[0].view(1,-1)), dim =0)
        pyg_data.edge_index =  torch.cat((pyg_data.edge_index, edge_inv), 1)
        pyg_data.edge_weight = torch.cat((pyg_data.edge_weight,pyg_data.edge_weight))
        pyg_data.edge_type = torch.cat((pyg_data.edge_type, pyg_data.edge_type))

        if name_test is not None:
            torch.save(pyg_data, f"{FOLDER}/homograph.pt")
        else:
            torch.save(pyg_data, f"{FOLDER}/homograph.pt")
        return pyg_data
    else:
        if name_test is not None:
            pyg_data = torch.load(f"{FOLDER}/homograph.pt")
        else:
            pyg_data = torch.load(f"{FOLDER}/homograph.pt")
        return pyg_data
