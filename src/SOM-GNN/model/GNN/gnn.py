import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GCNConv, RGCNConv, GatedGraphConv, GATv2Conv, GINConv
from transformers import BertModel


class FNSage(torch.nn.Module):
    def __init__(self, hidden_channels_1, hidden_channels_2, num_content_feature, num_classes, n_hop, dropout, seed):
        super(FNSage, self).__init__()
        torch.manual_seed(seed)
        self.n_hop = n_hop
        # Initialize the layers
        self.convs = nn.ModuleList()
        print(num_content_feature)
        self.convs.append(SAGEConv(num_content_feature, hidden_channels_1))
        for i in range (0,n_hop-2):
            self.convs.append(SAGEConv(hidden_channels_1, hidden_channels_1))
        self.convs.append(SAGEConv(hidden_channels_1, hidden_channels_2))
        self.dropout = dropout
        self.out = nn.Linear(hidden_channels_2, num_classes)

    def forward(self, x_content, edge_index, edge_type):
        
        # x_content_enc = self.post_enc(x_content)
        # x_style_content = self.style_enc(x_style)
        # x = torch.cat((x_content_enc, x_style_content),1)
        # # First Message Passing Layer (Transformation)
        x = x_content
        for i in range (0, self.n_hop):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Output layer 
        x = self.out(x)
        return x

class FNGCN(torch.nn.Module):
    def __init__(self, hidden_channels_1, hidden_channels_2, num_content_feature, num_classes, n_hop, dropout, seed):
        super(FNGCN, self).__init__()
        torch.manual_seed(seed)
        self.n_hop = n_hop
        # Initialize the layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(num_content_feature, hidden_channels_1))
        for i in range (0,n_hop-2):
            self.convs.append(GCNConv(hidden_channels_1, hidden_channels_1))
        self.convs.append(GCNConv(hidden_channels_1, hidden_channels_2))
        self.dropout = dropout
        self.out = nn.Linear(hidden_channels_2, num_classes)

    def forward(self, x_content, edge_index, edge_type):
        
        # x_content_enc = self.post_enc(x_content)
        # x_style_content = self.style_enc(x_style)
        # x = torch.cat((x_content_enc, x_style_content),1)
        # # First Message Passing Layer (Transformation)
        for i in range (self.n_hop):

            x = self.convs[i](x_content, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Output layer 
        x = self.out(x)
        return x

class FNRGCN(torch.nn.Module):
    def __init__(self, hidden_channels_1, hidden_channels_2, num_content_feature, num_classes, n_hop, dropout, seed):
        super(FNRGCN, self).__init__()
        torch.manual_seed(seed)
        self.n_hop = n_hop
        # Initialize the layers
        self.convs = nn.ModuleList()
        self.convs.append(RGCNConv(num_content_feature, hidden_channels_1, 3))
        for i in range (0,n_hop-2):
            self.convs.append(RGCNConv(hidden_channels_1, hidden_channels_1, 3))
        self.convs.append(RGCNConv(hidden_channels_1, hidden_channels_2, 3))
        self.dropout = dropout
        self.out = nn.Linear(hidden_channels_2, num_classes)

    def forward(self, x_content, edge_index, edge_type):
        
        # x_content_enc = self.post_enc(x_content)
        # x_style_content = self.style_enc(x_style)
        # x = torch.cat((x_content_enc, x_style_content),1)
        # # First Message Passing Layer (Transformation)
        for i in range (self.n_hop):

            x = self.convs[i](x_content, edge_index, edge_type)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Output layer 
        x = self.out(x)
        return x

